from typing import Tuple

import torch
from rainbowneko.ckpt_manager.format import CkptFormat
from torch.serialization import FILE_LIKE

class EmbFormat(CkptFormat):
    EXT = 'pt'

    def save_ckpt(self, sd_model: Tuple[str, torch.Tensor], save_f: FILE_LIKE):
        name, emb = sd_model
        torch.save({'string_to_param':{'*':emb}, 'name':name}, save_f)

    def load_ckpt(self, ckpt_f: FILE_LIKE, map_location="cpu"):
        state = torch.load(ckpt_f, map_location=map_location)
        if 'string_to_param' in state:
            emb = state['string_to_param']['*']
        else:
            emb = state['emb_params']
        emb.requires_grad_(False)
        return emb