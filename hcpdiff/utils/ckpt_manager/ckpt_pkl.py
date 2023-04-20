"""
ckpt_pkl.py
====================
    :Name:        save model with torch
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     8/04/2023
    :Licence:     MIT
"""

import torch
from torch import nn
import os
from hcpdiff.models.lora_base import LoraBlock, LoraGroup, split_state
from hcpdiff.utils.emb_utils import save_emb

class CkptManagerPKL:
    def __init__(self, lora_from_raw=False):
        self.lora_from_raw = lora_from_raw

    def set_save_dir(self, save_dir, emb_dir=None):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.emb_dir = emb_dir

    def exclude_state(self, state, key):
        if key is None:
            return state
        else:
            return {k:v for k,v in state.items() if key in k}

    def save_model(self, model: nn.Module, name, step, model_ema=None, exclude_key=None):
        sd_model = {
            'base': self.exclude_state(LoraBlock.extract_trainable_state_without_lora(model), exclude_key),
        }
        if model_ema is not None:
            sd_ema, sd_ema_lora = split_state(model_ema.state_dict())
            sd_model['base_ema'] = self.exclude_state(sd_ema, exclude_key)
        self._save_ckpt(sd_model, name, step)

    def save_model_with_lora(self, model: nn.Module, lora_blocks: LoraGroup, name, step, model_ema=None,
                             exclude_key=None):
        sd_model = {
            'base': self.exclude_state(LoraBlock.extract_trainable_state_without_lora(model), exclude_key),
        } if model is not None else {}
        if not lora_blocks.empty():
            sd_model['lora']=lora_blocks.state_dict(model if self.lora_from_raw else None)

        if model_ema is not None:
            sd_ema, sd_ema_lora = split_state(model_ema.state_dict())
            sd_model['base_ema'] = self.exclude_state(sd_ema, exclude_key)
            if not lora_blocks.empty():
                sd_model['lora_ema'] = {sd_ema_lora[k] for k in sd_model['lora'].keys()}

        self._save_ckpt(sd_model, name, step)

    def _save_ckpt(self, sd_model, name, step):
        save_path = os.path.join(self.save_dir, f"{name}-{step}.ckpt")
        torch.save(sd_model, save_path)

    def load_ckpt(self, ckpt_path, map_location='cpu'):
        return torch.load(ckpt_path, map_location=map_location)

    def load_ckpt_to_model(self, model:nn.Module, ckpt_path, model_ema=None):
        sd = self.load_ckpt(ckpt_path)
        if 'base' in sd:
            model.load_state_dict(sd['base'], strict=False)
        if 'lora' in sd:
            model.load_state_dict(sd['lora'], strict=False)

        if model_ema is not None:
            model_ema.load_state_dict(sd['base_ema'])
            model_ema.load_state_dict(sd['lora_ema'])


    def save_embedding(self, train_pts, step, replace):
        for k, v in train_pts.items():
            save_path = os.path.join(self.save_dir, f"{k}-{step}.pt")
            save_emb(save_path, v.data, replace=True)
            if replace:
                save_emb(f'{k}.pt', v.data, replace=True)