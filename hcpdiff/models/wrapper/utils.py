from dataclasses import dataclass
from rainbowneko.utils import is_dict

class TEHookCFG:
    def __init__(self, tokenizer_repeats: int = 1, clip_skip: int = 0, clip_final_norm: bool = True):
        self.tokenizer_repeats = tokenizer_repeats
        self.clip_skip = clip_skip
        self.clip_final_norm = clip_final_norm

    @classmethod
    def create(cls, cfg):
        if is_dict(cfg):
            return cls(**cfg)
        elif isinstance(cfg, cls):
            return cfg
        else:
            raise ValueError(f'Invalid TEHookCFG type: {type(cfg)}')
        
SD15_TEHookCFG = TEHookCFG()
SDXL_TEHookCFG = TEHookCFG(clip_skip=1, clip_final_norm=False)