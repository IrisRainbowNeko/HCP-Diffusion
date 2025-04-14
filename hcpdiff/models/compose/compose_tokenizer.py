"""
compose_tokenizer.py
====================
    :Name:        compose tokenizer
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     07/08/2023
    :Licence:     Apache-2.0

support for SDXL.
"""
from typing import Dict, Tuple, List

import torch
from transformers import AutoTokenizer, CLIPTokenizer, PreTrainedTokenizer, PretrainedConfig
from transformers.tokenization_utils_base import BatchEncoding

class ComposeTokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer_list: List[Tuple[str, CLIPTokenizer]], cat_dim=-1):
        self.cat_dim = cat_dim

        self.tokenizer_names = []
        for name, tokenizer in tokenizer_list:
            setattr(self, name, tokenizer)
            self.tokenizer_names.append(name)

        super().__init__()

        self.model_max_length = torch.tensor([tokenizer.model_max_length for name, tokenizer in tokenizer_list])

    @property
    def first_tokenizer(self):
        return getattr(self, self.tokenizer_names[0])

    @property
    def vocab_size(self):
        return len(self.first_tokenizer.encoder)

    @property
    def eos_token_id(self):
        return self.first_tokenizer.eos_token_id

    @property
    def bos_token_id(self):
        return self.first_tokenizer.bos_token_id

    def get_vocab(self):
        return self.first_tokenizer.get_vocab()

    def tokenize(self, text, **kwargs) -> List[str]:
        return self.first_tokenizer.tokenize(text, **kwargs)

    def add_tokens( self, new_tokens, special_tokens: bool = False) -> List[int]:
        return [getattr(self, name).add_tokens(new_tokens, special_tokens) for name in self.tokenizer_names]
    
    def save_vocabulary(self, save_directory: str, filename_prefix = None) -> Tuple[str]:
        return self.first_tokenizer.save_vocabulary(save_directory, filename_prefix)

    def __call__(self, text, *args, max_length=None, **kwargs):
        if isinstance(max_length, torch.Tensor):
            token_list: List[BatchEncoding] = [getattr(self, name)(text, *args, max_length=max_length_i, **kwargs)
                for name, max_length_i in zip(self.tokenizer_names, max_length)]
        else:
            token_list: List[BatchEncoding] = [getattr(self, name)(text, *args, max_length=max_length, **kwargs) for name in self.tokenizer_names]

        input_ids = torch.cat([token.input_ids for token in token_list], dim=-1)  # [N_tokenizer, N_token]
        attention_mask = torch.cat([token.attention_mask for token in token_list], dim=-1)
        return BatchEncoding({'input_ids':input_ids, 'attention_mask':attention_mask})

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: List[Tuple[str, str]], *args,
                        subfolder: Dict[str, str] = None, revision: str = None, **kwargs):
        tokenizer_list = [(name, AutoTokenizer.from_pretrained(path, subfolder=subfolder[name], **kwargs)) for name, path in pretrained_model_name_or_path]
        compose_tokenizer = cls(tokenizer_list)
        return compose_tokenizer
