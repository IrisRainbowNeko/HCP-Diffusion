import os
from typing import Dict, Union, Tuple, List

import torch
from loguru import logger
from torch import nn

from .compose_textencoder import ComposeTextEncoder
from ..text_emb_ex import EmbeddingPTHook
from ..textencoder_ex import TEEXHook
from ...utils.net_utils import load_emb
from ..container import ParameterGroup

class ComposeEmbPTHook(nn.Module):
    def __init__(self, hook_list: List[Tuple[str, EmbeddingPTHook]]):
        super().__init__()
        self.hook_list = hook_list
        self.emb_train = nn.ParameterList()

    @property
    def N_repeats(self):
        return self.hook_list[0][1].N_repeats

    @N_repeats.setter
    def N_repeats(self, value):
        for name, hook in self.hook_list:
            hook.N_repeats = value

    def add_emb(self, emb: nn.Parameter, token_id_list: List[int]):
        emb_len = 0
        # Same word in different tokenizer may have different token_id
        for (name, hook), token_id in zip(self.hook_list, token_id_list):
            hook.add_emb(emb[:, emb_len:emb_len+hook.embedding_dim], token_id)
            emb_len += hook.embedding_dim

    def remove(self):
        for name, hook in self.hook_list:
            hook.remove()

    @classmethod
    def hook(cls, ex_words_emb: Dict[str, ParameterGroup], tokenizer, text_encoder, log=False, **kwargs):
        if isinstance(text_encoder, ComposeTextEncoder):
            hook_list = []

            emb_len = 0
            for i, (name, tokenizer_i) in enumerate(tokenizer.tokenizer_list):
                text_encoder_i = getattr(text_encoder, name)
                if log:
                    logger.info(f'compose hook: {name}')
                embedding_dim = text_encoder_i.get_input_embeddings().embedding_dim
                ex_words_emb_i = {k:v[i] for k, v in ex_words_emb.items()}
                emb_len += embedding_dim
                hook_list.append((name, EmbeddingPTHook.hook(ex_words_emb_i, tokenizer_i, text_encoder_i, log=log, **kwargs)))

            return cls(hook_list)
        else:
            return EmbeddingPTHook.hook(ex_words_emb, tokenizer, text_encoder, log, **kwargs)

    @classmethod
    def hook_from_dir(cls, emb_dir, tokenizer, text_encoder, log=True, device='cuda:0', **kwargs) -> Union[
        Tuple['ComposeEmbPTHook', Dict], Tuple[EmbeddingPTHook, Dict]]:
        if isinstance(text_encoder, ComposeTextEncoder):
            # multi text encoder
            #ex_words_emb = {file[:-3]:load_emb(os.path.join(emb_dir, file)).to(device) for file in os.listdir(emb_dir) if file.endswith('.pt')}

            # slice of nn.Parameter cannot return grad. Split the tensor
            ex_words_emb = {}
            emb_dims = [x.embedding_dim for x in text_encoder.get_input_embeddings()]
            for file in os.listdir(emb_dir):
                if file.endswith('.pt'):
                    emb = load_emb(os.path.join(emb_dir, file)).to(device)
                    emb = ParameterGroup([nn.Parameter(item, requires_grad=False) for item in emb.split(emb_dims, dim=1)])
                    ex_words_emb[file[:-3]] = emb
            return cls.hook(ex_words_emb, tokenizer, text_encoder, log, **kwargs), ex_words_emb
        else:
            return EmbeddingPTHook.hook_from_dir(emb_dir, tokenizer, text_encoder, log, device, **kwargs)

class ComposeTEEXHook:
    def __init__(self, tehook_list: List[Tuple[str, TEEXHook]], cat_dim=-1):
        self.tehook_list = tehook_list
        self.cat_dim = cat_dim

    @property
    def N_repeats(self):
        return self.tehook_list[0][1].N_repeats

    @N_repeats.setter
    def N_repeats(self, value):
        for name, tehook in self.tehook_list:
            tehook.N_repeats = value

    @property
    def clip_skip(self):
        return self.tehook_list[0][1].clip_skip

    @clip_skip.setter
    def clip_skip(self, value):
        for name, tehook in self.tehook_list:
            tehook.clip_skip = value

    def encode_prompt_to_emb(self, prompt):
        emb_list = [tehook.encode_prompt_to_emb(prompt) for name, tehook in self.tehook_list]
        encoder_hidden_states, pooled_output = list(zip(*emb_list))
        return torch.cat(encoder_hidden_states, dim=self.cat_dim), pooled_output

    def enable_xformers(self):
        for name, tehook in self.tehook_list:
            tehook.enable_xformers()

    @staticmethod
    def mult_attn(prompt_embeds, attn_mult):
        return TEEXHook.mult_attn(prompt_embeds, attn_mult)

    @classmethod
    def hook(cls, text_enc: nn.Module, tokenizer, N_repeats=3, clip_skip=0, clip_final_norm=True, device='cuda') -> Union['ComposeTEEXHook', TEEXHook]:
        if isinstance(text_enc, ComposeTextEncoder):
            # multi text encoder
            tehook_list = [(name, TEEXHook.hook(getattr(text_enc, name), tokenizer_i, N_repeats, clip_skip, clip_final_norm, device))
                for name, tokenizer_i in tokenizer.tokenizer_list]
            return cls(tehook_list)
        else:
            # single text encoder
            return TEEXHook.hook(text_enc, tokenizer, N_repeats, clip_skip, device)

    @classmethod
    def hook_pipe(cls, pipe, N_repeats=3, clip_skip=0, clip_final_norm=True):
        return cls.hook(pipe.text_encoder, pipe.tokenizer, N_repeats=N_repeats, device='cuda', clip_skip=clip_skip, clip_final_norm=clip_final_norm)
