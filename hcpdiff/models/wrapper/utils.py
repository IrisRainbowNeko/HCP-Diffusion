from dataclasses import dataclass
from rainbowneko.utils import is_dict

@dataclass
class TEHookCFG:
    tokenizer_repeats = 1
    clip_skip = 0
    clip_final_norm = True

    @classmethod
    def create(cls, cfg):
        if is_dict(cfg):
            return cls(**cfg)
        elif isinstance(cfg, cls):
            return cfg
        else:
            raise ValueError(f'Invalid TEHookCFG type: {type(cfg)}')