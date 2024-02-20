from typing import List, Union

import torch
from hcpdiff.models import TokenizerHook
from hcpdiff.models.compose import ComposeTEEXHook, ComposeEmbPTHook
from hcpdiff.utils.net_utils import get_dtype, to_cpu, to_cuda
from torch.cuda.amp import autocast

from .base import BasicAction, from_memory_context, MemoryMixin

class TextHookAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, TE=None, tokenizer=None, emb_dir: str = 'embs/', N_repeats: int = 1, layer_skip: int = 0, TE_final_norm: bool = True):
        super().__init__()
        self.TE = TE
        self.tokenizer = tokenizer

        self.emb_dir = emb_dir
        self.N_repeats = N_repeats
        self.layer_skip = layer_skip
        self.TE_final_norm = TE_final_norm

    def forward(self, memory, **states):
        self.TE = self.TE or memory.text_encoder
        self.tokenizer = self.tokenizer or memory.tokenizer

        memory.emb_hook, _ = ComposeEmbPTHook.hook_from_dir(self.emb_dir, self.tokenizer, self.TE, N_repeats=self.N_repeats)
        memory.te_hook = ComposeTEEXHook.hook(self.TE, self.tokenizer, N_repeats=self.N_repeats, device='cuda',
                                              clip_skip=self.layer_skip, clip_final_norm=self.TE_final_norm)
        memory.token_ex = TokenizerHook(self.tokenizer)
        return states

class TextEncodeAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, prompt: Union[List, str], negative_prompt: Union[List, str], bs: int = None, te_hook=None):
        super().__init__()
        if isinstance(prompt, str) and bs is not None:
            prompt = [prompt]*bs
            negative_prompt = [negative_prompt]*bs

        self.prompt = prompt
        self.negative_prompt = negative_prompt

        self.te_hook = te_hook

    def forward(self, memory, dtype: str, device, amp=None, **states):
        te_hook = self.te_hook or memory.te_hook
        with autocast(enabled=amp is not None, dtype=get_dtype(amp)):
            emb, pooled_output = te_hook.encode_prompt_to_emb(self.negative_prompt+self.prompt)
            # emb = emb.to(dtype=get_dtype(dtype), device=device)
        return {**states, 'prompt':self.prompt, 'negative_prompt':self.negative_prompt, 'prompt_embeds':emb, 'amp':amp,
            'device':device, 'dtype':dtype}

class AttnMultTextEncodeAction(TextEncodeAction):
    @from_memory_context
    def __init__(self, prompt: Union[List, str], negative_prompt: Union[List, str], bs: int = None, te_hook=None, token_ex=None):
        super().__init__(prompt, negative_prompt, bs, te_hook)
        self.token_ex = token_ex

    def forward(self, memory, dtype: str, device, amp=None, **states):
        te_hook = self.te_hook or memory.te_hook
        token_ex = self.token_ex or memory.token_ex

        offload = memory.text_encoder.device.type == 'cpu'
        if offload:
            to_cuda(memory.text_encoder)

        mult_p, clean_text_p = token_ex.parse_attn_mult(self.prompt)
        mult_n, clean_text_n = token_ex.parse_attn_mult(self.negative_prompt)
        with autocast(enabled=amp is not None, dtype=get_dtype(amp)):
            emb, pooled_output, attention_mask = te_hook.encode_prompt_to_emb(clean_text_n+clean_text_p)
            emb_n, emb_p = emb.chunk(2)
        emb_p = te_hook.mult_attn(emb_p, mult_p)
        emb_n = te_hook.mult_attn(emb_n, mult_n)

        if offload:
            to_cpu(memory.text_encoder)

        return {**states, 'prompt':self.prompt, 'negative_prompt':self.negative_prompt, 'prompt_embeds':torch.cat([emb_n, emb_p], dim=0),
            'device':device, 'dtype':dtype, 'amp':amp, 'encoder_attention_mask':attention_mask}
