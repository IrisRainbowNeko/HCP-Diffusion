from typing import Dict, Tuple, List
from rainbowneko.utils import Path_Like
from hcpdiff.models.compose import ComposeEmbPTHook
from torch import Tensor, nn

class CfgEmbPTParser:
    def __init__(self, emb_dir: Path_Like, cfg_pt: Dict[str, Dict], lr: float = 1e-5, weight_decay: float = 0):
        self.emb_dir = emb_dir
        self.cfg_pt = cfg_pt
        self.lr = lr
        self.weight_decay = weight_decay

    def get_params_group(self, model) -> Tuple[List, Dict[str, Tensor]]:
        self.embedding_hook, self.ex_words_emb = ComposeEmbPTHook.hook_from_dir(
            self.emb_dir, model.tokenizer, model.TE, N_repeats=model.tokenizer.N_repeats)
        self.embedding_hook.requires_grad_(False)

        train_params_emb = []
        train_pts = {}
        for pt_name, info in self.cfg_pt.items():
            word_emb: nn.Parameter | nn.ParameterDict = self.ex_words_emb[pt_name]
            train_pts[pt_name] = word_emb
            word_emb.requires_grad_(True)
            self.embedding_hook.emb_train.append(word_emb)
            param_group = {'params':word_emb.parameters() if hasattr(word_emb, 'parameters') else [word_emb]}
            if 'lr' in info:
                param_group['lr'] = info.lr
            if 'weight_decay' in info:
                param_group['weight_decay'] = info.weight_decay
            train_params_emb.append(param_group)

        return train_params_emb, train_pts
