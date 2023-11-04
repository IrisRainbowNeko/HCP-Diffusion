from typing import List, Union

from torch.cuda.amp import autocast

from hcpdiff.models import TokenizerHook
from hcpdiff.models.compose import ComposeTEEXHook, ComposeEmbPTHook
from .base import BasicAction, from_memory_context

class TextHookAction(BasicAction):
    @from_memory_context
    def __init__(self, TE=None, tokenizer=None, emb_dir: str = 'embs/', N_repeats: int = 1, layer_skip: int = 0, TE_final_norm: bool = True):
        super().__init__()
        self.TE = TE
        self.tokenizer = tokenizer

        self.emb_dir = emb_dir
        self.N_repeats = N_repeats
        self.layer_skip = layer_skip
        self.TE_final_norm = TE_final_norm

    def forward(self, **states):
        emb_hook, _ = ComposeEmbPTHook.hook_from_dir(self.emb_dir, self.tokenizer, self.TE, N_repeats=self.N_repeats)
        te_hook = ComposeTEEXHook.hook(self.TE, self.tokenizer, N_repeats=self.N_repeats, device='cuda',
                                       clip_skip=self.layer_skip, clip_final_norm=self.TE_final_norm)
        token_ex = TokenizerHook(self.tokenizer)
        return {'emb_hook':emb_hook, 'te_hook':te_hook, 'token_ex':token_ex, **states}

class TextEncodeAction(BasicAction):
    @from_memory_context
    def __init__(self, prompt: Union[List, str], negative_prompt: Union[List, str], bs: int = None, dtype: str = 'fp32'):
        super().__init__()
        if bs is not None:
            prompt = [prompt]*bs
            negative_prompt = [negative_prompt]*bs

        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.dtype = dtype

    def forward(self, te_hook, dtype, device, **states):
        with autocast(enabled=self.dtype == 'amp'):
            emb, pooled_output = te_hook.encode_prompt_to_emb(self.negative_prompt+self.prompt)
            emb = emb.to(dtype=dtype, device=device)
            emb_n, emb_p = emb.chunk(2)
        return {'emb_n':emb_n, 'emb_p':emb_p, **states}

class AttnMultTextEncodeAction(TextEncodeAction):
    def forward(self, te_hook, token_ex, dtype, device, **states):
        mult_p, clean_text_p = token_ex.parse_attn_mult(self.prompt)
        mult_n, clean_text_n = token_ex.parse_attn_mult(self.negative_prompt)
        with autocast(enabled=self.dtype == 'amp'):
            emb, pooled_output = te_hook.encode_prompt_to_emb(clean_text_n+clean_text_p)
            emb = emb.to(dtype=dtype, device=device)
            emb_n, emb_p = emb.chunk(2)
        emb_p = te_hook.mult_attn(emb_p, mult_p)
        emb_n = te_hook.mult_attn(emb_n, mult_n)
        return {'emb_n':emb_n, 'emb_p':emb_p, **states}
