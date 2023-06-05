"""
tokenizer_ex.py
====================
    :Name:        extend Tokenizer
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import torch
from collections import deque

class TokenizerHook:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer=tokenizer

    @staticmethod
    def get_mult_list(text):
        """
            text: ... {sentence part that need change attention:attention multiply} ...
        """
        out_str = ""
        clean_str = ""
        mult_stack = deque()
        mult_list = deque()

        i = len(text) - 1
        while i >= 0:
            c = text[i]
            if c == '{':
                mult = mult_stack.pop()
                mult_list.appendleft(mult)
                out_str = c + out_str
            elif c == '}':
                idx = i - 1
                while not (text[idx] == '{' or text[idx] == '}' or text[idx] == ':'):
                    idx -= 1
                if text[idx] == ':':
                    mult = float(text[idx + 1:i])
                    out_str = c + out_str
                    i = idx
                else:
                    mult = 1.1
                    out_str = text[idx + 1:i + 1] + out_str
                    clean_str = text[idx + 1:i] + clean_str
                    i = idx + 1
                mult_stack.append(mult)
                mult_list.appendleft(1 / mult)
            else:
                out_str = c + out_str
                clean_str = c + clean_str
            i -= 1
        return mult_list, out_str, clean_str

    def parse_attn_mult_one(self, text: str):
        if text is None or len(text)==0:
            return torch.tensor([]), ""
        mult_list, out_str, clean_str = self.get_mult_list(text)
        mult_iter = iter(mult_list)
        token_seq = self.tokenizer.tokenize(out_str)
        mult = 1.0
        mult_seq = []
        for token in token_seq:
            if token[0] == '{' or token[0] == '}':
                mult *= next(mult_iter)
            else:
                mult_seq.append(mult)
        return torch.tensor(mult_seq), clean_str

    def parse_attn_mult(self, text):
        if isinstance(text, str):
            mult_seq, clean_str = self.parse_attn_mult_one(text)
            return [mult_seq], [clean_str]
        else:
            return list(zip(*[self.parse_attn_mult_one(item) for item in text]))
