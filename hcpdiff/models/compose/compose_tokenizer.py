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
from rainbowneko.utils import BatchableDict

class ComposeTokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizers: Dict[str, CLIPTokenizer]):

        self.tokenizer_names = []
        for name, tokenizer in tokenizers.items():
            setattr(self, name, tokenizer)
            self.tokenizer_names.append(name)

        super().__init__()

        # self.model_max_length = torch.tensor([tokenizer.model_max_length for name, tokenizer in tokenizer_list])
        self.model_max_length = {name: tokenizer.model_max_length for name, tokenizer in tokenizers.items()}

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
        if isinstance(max_length, dict):
            token_infos: Dict[str, BatchEncoding] = {name: getattr(self, name)(text, *args, max_length=max_length[name], **kwargs)
                for name in self.tokenizer_names}
        else:
            token_infos: Dict[str, BatchEncoding] = {name: getattr(self, name)(text, *args, max_length=max_length, **kwargs)
                for name in self.tokenizer_names}

        input_ids = BatchableDict({name: token.input_ids for name, token in token_infos.items()})  # [N_tokenizer, N_token]
        attention_mask = BatchableDict({name: token.get('attention_mask', None) for name, token in token_infos.items()})
        position_ids = BatchableDict({name: token.get('position_ids', None) for name, token in token_infos.items()})
        return BatchEncoding({'input_ids':input_ids, 'attention_mask':attention_mask, 'position_ids':position_ids})

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: List[Tuple[str, str]], *args,
                        subfolder: Dict[str, str] = None, revision: str = None, **kwargs):
        tokenizer_list = [(name, AutoTokenizer.from_pretrained(path, subfolder=subfolder[name], **kwargs)) for name, path in pretrained_model_name_or_path]
        compose_tokenizer = cls(tokenizer_list)
        return compose_tokenizer

    def __repr__(self):
        return f'ComposeTokenizer(\n' + '\n'.join([f'  {name}: {repr(getattr(self, name))}' for name in self.tokenizer_names]) + ')'

    @staticmethod
    def tokenize_ex(tokenizer, *args, device='cpu', squeeze=False, **kwargs):
        if isinstance(tokenizer, ComposeTokenizer):
            max_length = {name: (tok := getattr(tokenizer, name)).model_max_length * getattr(tok, 'N_repeats', 1) for name in tokenizer.tokenizer_names}
        else:
            max_length = tokenizer.model_max_length * getattr(tokenizer, 'N_repeats', 1)

        text_inputs = tokenizer(
            *args,
            max_length=max_length,
            **kwargs
        )

        def proc_tensor(v):
            if v is None:
                return None
            elif squeeze:
                return v.squeeze().to(device)
            else:
                return v.to(device)

        for k, v in text_inputs.items():
            if torch.is_tensor(v):
                text_inputs[k] = proc_tensor(v)
            elif isinstance(v, (dict, BatchableDict)):
                for name, vi in v.items():
                    if torch.is_tensor(vi):
                        v[name] = proc_tensor(vi)

        return text_inputs
