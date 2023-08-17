import os
from typing import Dict, Union, Tuple, List

import torch
from loguru import logger
from torch import nn

from .compose_textencoder import ComposeTextEncoder
from ..text_emb_ex import EmbeddingPTHook
from ..textencoder_ex import TEEXHook
from ...utils.net_utils import load_emb

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
    def hook(cls, ex_words_emb: Dict[str, nn.Parameter], tokenizer, text_encoder, log=False, **kwargs):
        if isinstance(text_encoder, ComposeTextEncoder):
            hook_list = []

            emb_len = 0
            for (name, tokenizer_i), (_, text_encoder_i) in zip(tokenizer, text_encoder):
                if log:
                    logger.info(f'compose hook: {name}')
                embedding_dim = text_encoder_i.text_model.embeddings.token_embedding.embedding_dim
                ex_words_emb_i = {k:v[emb_len:emb_len+embedding_dim] for k, v in ex_words_emb.items()}
                hook_list.append(EmbeddingPTHook.hook(ex_words_emb_i, tokenizer_i, text_encoder_i, log=log, **kwargs))

            return cls(hook_list)
        else:
            return EmbeddingPTHook.hook(ex_words_emb, tokenizer, text_encoder, log, **kwargs)

    @classmethod
    def hook_from_dir(cls, emb_dir, tokenizer, text_encoder, log=True, device='cuda:0', **kwargs) -> Union[
        Tuple['ComposeEmbPTHook', Dict], Tuple[EmbeddingPTHook, Dict]]:
        if isinstance(text_encoder, ComposeTextEncoder):
            # multi text encoder
            ex_words_emb = {file[:-3]:nn.Parameter(load_emb(os.path.join(emb_dir, file)).to(device), requires_grad=False)
                for file in os.listdir(emb_dir) if file.endswith('.pt')}
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
        return torch.cat(emb_list, dim=self.cat_dim)

    def enable_xformers(self):
        for name, tehook in self.tehook_list:
            tehook.enable_xformers()

    @classmethod
    def hook(cls, text_enc: nn.Module, tokenizer, N_repeats=3, clip_skip=0, device='cuda') -> Union['ComposeTEEXHook', TEEXHook]:
        if isinstance(text_enc, ComposeTextEncoder):
            # multi text encoder
            tehook_list = [TEEXHook.hook(getattr(text_enc, name), tokenizer_i, N_repeats, clip_skip, device)
                for name, (_, tokenizer_i) in zip(text_enc.model_names, tokenizer)]
            return cls(tehook_list)
        else:
            # single text encoder
            return TEEXHook.hook(text_enc, tokenizer, N_repeats, clip_skip, device)
