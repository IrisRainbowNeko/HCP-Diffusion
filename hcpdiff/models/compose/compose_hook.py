from pathlib import Path
from typing import Dict, Union, Tuple

import torch
from torch import nn

from hcpdiff.utils.net_utils import load_emb
from .compose_textencoder import ComposeTextEncoder
from ..text_emb_ex import EmbeddingPTHook
from ..textencoder_ex import TEEXHook

class ComposeEmbPTHook(nn.Module):
    def __init__(self, hooks: Dict[str, EmbeddingPTHook]):
        super().__init__()
        self.hooks = hooks
        self.emb_train = nn.ParameterList() # [ParameterDict{model_name:Parameter, ...}, ...]

    @property
    def N_repeats(self):
        return {name:hook.N_repeats for name, hook in self.hooks.items()}

    @N_repeats.setter
    def N_repeats(self, value):
        for name, hook in self.hooks.items():
            if isinstance(value, int):
                hook.N_repeats = value
            else:
                hook.N_repeats = value[name]

    def add_emb(self, emb: Dict[str, nn.Parameter], token_ids: Dict[str, int]):
        # Same word in different tokenizer may have different token_id
        for name, hook in self.hooks.items():
            hook.add_emb(emb[name], token_ids[name])

    def remove(self):
        for name, hook in self.hooks.items():
            hook.remove()

    @classmethod
    def hook(cls, ex_words_emb: Dict[str, nn.ParameterDict], tokenizer, text_encoder, **kwargs):
        if isinstance(text_encoder, ComposeTextEncoder):
            hooks = {}

            emb_len = 0
            for name in tokenizer.tokenizer_names:
                text_encoder_i = getattr(text_encoder, name)
                tokenizer_i = getattr(tokenizer, name)
                embedding_dim = text_encoder_i.get_input_embeddings().embedding_dim
                ex_words_emb_i = {k:v[name] for k, v in ex_words_emb.items()}  # {word_name:Parameter, ...}
                emb_len += embedding_dim
                hooks[name] = EmbeddingPTHook.hook(ex_words_emb_i, tokenizer_i, text_encoder_i, **kwargs)

            return cls(hooks)
        else:
            return EmbeddingPTHook.hook(ex_words_emb, tokenizer, text_encoder, **kwargs)

    @classmethod
    def hook_from_dir(cls, emb_dir: str | Path, tokenizer, text_encoder, device='cuda', **kwargs) -> (
            Tuple['ComposeEmbPTHook', Dict[str, nn.ParameterDict]] | Tuple[EmbeddingPTHook, Dict[str, nn.Parameter]]):
        emb_dir = Path(emb_dir) if emb_dir is not None else None
        if isinstance(text_encoder, ComposeTextEncoder):
            # multi text encoder
            ex_words_emb = {}  # {word_name:{model_name:Tensor, ...}, ...}
            if emb_dir is not None and emb_dir.exists():
                for file in emb_dir.glob('*.pt'):
                    emb = load_emb(file)  # {model_name:Tensor, ...}
                    emb = nn.ParameterDict({name:nn.Parameter(emb_i.to(device), requires_grad=False) for name, emb_i in emb.items()})
                    ex_words_emb[file.stem] = emb
            return cls.hook(ex_words_emb, tokenizer, text_encoder, **kwargs), ex_words_emb
        else:
            return EmbeddingPTHook.hook_from_dir(emb_dir, tokenizer, text_encoder, **kwargs)

class ComposeTEEXHook:
    def __init__(self, tehooks: Dict[str, TEEXHook]):
        self.tehooks = tehooks

    @property
    def N_repeats(self):
        return {name:tehook.N_repeats for name, tehook in self.tehooks.items()}

    @N_repeats.setter
    def N_repeats(self, value: int | Dict[str, int]):
        for name, tehook in self.tehooks.items():
            if isinstance(value, int):
                tehook.N_repeats = value
            else:
                tehook.N_repeats = value[name]

    @property
    def clip_skip(self):
        return {name:tehook.clip_skip for name, tehook in self.tehooks.items()}

    @clip_skip.setter
    def clip_skip(self, value: int | Dict[str, int]):
        for name, tehook in self.tehooks.items():
            if isinstance(value, int):
                tehook.clip_skip = value
            else:
                tehook.clip_skip = value[name]

    @property
    def clip_final_norm(self):
        return {name:tehook.clip_final_norm for name, tehook in self.tehooks.items()}

    @clip_final_norm.setter
    def clip_final_norm(self, value: bool | Dict[str, bool]):
        for name, tehook in self.tehooks.items():
            if isinstance(value, bool):
                tehook.clip_final_norm = value
            else:
                tehook.clip_final_norm = value[name]

    @property
    def use_attention_mask(self):
        return {name:tehook.use_attention_mask for name, tehook in self.tehooks.items()}

    @use_attention_mask.setter
    def use_attention_mask(self, value: bool | Dict[str, bool]):
        for name, tehook in self.tehooks.items():
            if isinstance(value, bool):
                tehook.use_attention_mask = value
            else:
                tehook.use_attention_mask = value[name]

    def enable_xformers(self):
        for name, tehook in self.tehooks.items():
            tehook.enable_xformers()

    @staticmethod
    def mult_attn(prompt_embeds, attn_mult):
        return TEEXHook.mult_attn(prompt_embeds, attn_mult)

    @classmethod
    def hook(cls, text_enc: nn.Module, tokenizer, N_repeats=1, clip_skip=0, clip_final_norm=True, use_attention_mask=False) -> Union[
        'ComposeTEEXHook', TEEXHook]:
        if isinstance(text_enc, ComposeTextEncoder):
            # multi text encoder
            get_data = lambda name, data:data[name] if isinstance(data, dict) else data
            tehooks = {name:TEEXHook.hook(getattr(text_enc, name), getattr(tokenizer, name), get_data(name, N_repeats), get_data(name, clip_skip),
                                          get_data(name, clip_final_norm), use_attention_mask=get_data(name, use_attention_mask))
                for name in tokenizer.tokenizer_names}
            return cls(tehooks)
        else:
            # single text encoder
            return TEEXHook.hook(text_enc, tokenizer, N_repeats, clip_skip, clip_final_norm, use_attention_mask=use_attention_mask)

    @classmethod
    def hook_pipe(cls, pipe, N_repeats=1, clip_skip=0, clip_final_norm=True, use_attention_mask=False):
        return cls.hook(pipe.text_encoder, pipe.tokenizer, N_repeats=N_repeats, clip_skip=clip_skip, clip_final_norm=clip_final_norm,
                        use_attention_mask=use_attention_mask)
