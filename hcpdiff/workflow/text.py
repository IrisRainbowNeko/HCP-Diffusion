from typing import List, Union

import torch
from hcpdiff.models import TokenizerHook
from hcpdiff.models.compose import ComposeTEEXHook, ComposeEmbPTHook
from hcpdiff.utils import pad_attn_bias
from hcpdiff.utils.net_utils import get_dtype, to_cpu, to_cuda
from rainbowneko.infer import BasicAction
from torch.cuda.amp import autocast

class TextHookAction(BasicAction):
    def __init__(self, emb_dir: str = None, N_repeats: int = 1, layer_skip: int = 0, TE_final_norm: bool = True,
                 use_attention_mask=False, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)

        self.emb_dir = emb_dir
        self.N_repeats = N_repeats
        self.layer_skip = layer_skip
        self.TE_final_norm = TE_final_norm
        self.use_attention_mask = use_attention_mask

    def forward(self, TE, tokenizer, in_preview=False, te_hook:ComposeTEEXHook=None, emb_hook=None, **states):
        if in_preview and emb_hook is not None:
            emb_hook.N_repeats = self.N_repeats
        else:
            emb_hook, _ = ComposeEmbPTHook.hook_from_dir(self.emb_dir, tokenizer, TE, N_repeats=self.N_repeats)
            tokenizer.N_repeats = self.N_repeats

        if in_preview:
            te_hook.N_repeats = self.N_repeats
            te_hook.clip_skip = self.layer_skip
            te_hook.clip_final_norm = self.TE_final_norm
            te_hook.use_attention_mask = self.use_attention_mask
        else:
            te_hook = ComposeTEEXHook.hook(TE, tokenizer, N_repeats=self.N_repeats,
                                       clip_skip=self.layer_skip, clip_final_norm=self.TE_final_norm, use_attention_mask=self.use_attention_mask)
        token_ex = TokenizerHook(tokenizer)
        return {'te_hook':te_hook, 'emb_hook':emb_hook, 'token_ex':token_ex}

class TextEncodeAction(BasicAction):
    def __init__(self, prompt: Union[List, str], negative_prompt: Union[List, str], bs: int = None, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        if isinstance(prompt, str) and bs is not None:
            prompt = [prompt]*bs
            negative_prompt = [negative_prompt]*bs

        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.bs = bs

    def forward(self, te_hook, TE, dtype: str, device, amp=None, prompt=None, negative_prompt=None, model_offload=False, **states):
        prompt = prompt or self.prompt
        negative_prompt = negative_prompt or self.negative_prompt

        if model_offload:
            to_cuda(TE)

        with autocast(enabled=amp is not None, dtype=get_dtype(amp)):
            emb, pooled_output, attention_mask = te_hook.encode_prompt_to_emb(negative_prompt+prompt)
            if attention_mask is not None:
                emb, attention_mask = pad_attn_bias(emb, attention_mask)

        if model_offload:
            to_cpu(TE)

        if not isinstance(te_hook, ComposeTEEXHook):
            pooled_output = None
        return {'prompt':prompt, 'negative_prompt':negative_prompt, 'prompt_embeds':emb, 'encoder_attention_mask':attention_mask,
            'pooled_output':pooled_output}

class AttnMultTextEncodeAction(TextEncodeAction):
    def forward(self, te_hook, token_ex, TE, dtype: str, device, amp=None, prompt=None, negative_prompt=None, model_offload=False, **states):
        prompt = prompt or self.prompt
        negative_prompt = negative_prompt or self.negative_prompt

        if model_offload:
            to_cuda(TE)

        mult_p, clean_text_p = token_ex.parse_attn_mult(prompt)
        mult_n, clean_text_n = token_ex.parse_attn_mult(negative_prompt)
        with autocast(enabled=amp is not None, dtype=get_dtype(amp)):
            emb, pooled_output, attention_mask = te_hook.encode_prompt_to_emb(clean_text_n+clean_text_p)
            if attention_mask is not None:
                emb, attention_mask = pad_attn_bias(emb, attention_mask)
            emb_n, emb_p = emb.chunk(2)
        emb_p = te_hook.mult_attn(emb_p, mult_p)
        emb_n = te_hook.mult_attn(emb_n, mult_n)

        if model_offload:
            to_cpu(TE)

        return {'prompt':list(clean_text_p), 'negative_prompt':list(clean_text_n), 'prompt_embeds':torch.cat([emb_n, emb_p], dim=0),
            'encoder_attention_mask':attention_mask, 'pooled_output':pooled_output}
