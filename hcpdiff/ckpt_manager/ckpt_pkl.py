"""
ckpt_pkl.py
====================
    :Name:        save model with torch
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     8/04/2023
    :Licence:     MIT
"""

from typing import Dict
import os

import torch
from torch import nn

from hcpdiff.models.lora_base import LoraBlock, LoraGroup, split_state
from hcpdiff.models.plugin import PluginGroup, BasePluginBlock
from hcpdiff.utils.net_utils import save_emb
from .base import CkptManagerBase

class CkptManagerPKL(CkptManagerBase):
    def __init__(self, plugin_from_raw=False, **kwargs):
        self.plugin_from_raw = plugin_from_raw

    def set_save_dir(self, save_dir, emb_dir=None):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.emb_dir = emb_dir

    def exclude_state(self, state, key):
        if key is None:
            return state
        else:
            return {k: v for k, v in state.items() if key not in k}

    def save_model(self, model: nn.Module, name, step, model_ema=None, exclude_key=None):
        sd_model = {
            'base': self.exclude_state(LoraBlock.extract_trainable_state_without_lora(model), exclude_key),
        }
        if model_ema is not None:
            sd_ema, sd_ema_lora = split_state(model_ema.state_dict())
            sd_model['base_ema'] = self.exclude_state(sd_ema, exclude_key)
        self._save_ckpt(sd_model, name, step)

    def save_plugins(self, host_model: nn.Module, plugins: Dict[str, PluginGroup], name:str, step:int, model_ema=None):
        if len(plugins)>0:
            sd_plugin={}
            for plugin_name, plugin in plugins.items():
                sd_plugin['plugin'] = plugin.state_dict(host_model if self.plugin_from_raw else None)
                if model_ema is not None:
                    sd_plugin['plugin_ema'] = plugin.state_dict(model_ema)
                self._save_ckpt(sd_plugin, f'{name}-{plugin_name}', step)

    def save_model_with_lora(self, model: nn.Module, lora_blocks: LoraGroup, name:str, step:int, model_ema=None,
                             exclude_key=None):
        sd_model = {
            'base': self.exclude_state(BasePluginBlock.extract_state_without_plugin(model, trainable=True), exclude_key),
        } if model is not None else {}
        if (lora_blocks is not None) and (not lora_blocks.empty()):
            sd_model['lora'] = lora_blocks.state_dict(model if self.plugin_from_raw else None)

        if model_ema is not None:
            ema_state = model_ema.state_dict()
            if model is not None:
                sd_ema = {k:ema_state[k] for k in sd_model['base'].keys()}
                sd_model['base_ema'] = self.exclude_state(sd_ema, exclude_key)
            if (lora_blocks is not None) and (not lora_blocks.empty()):
                sd_model['lora_ema'] = lora_blocks.state_dict(model_ema)

        self._save_ckpt(sd_model, name, step)

    def _save_ckpt(self, sd_model, name=None, step=None, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{name}-{step}.ckpt")
        torch.save(sd_model, save_path)

    def load_ckpt(self, ckpt_path, map_location='cpu'):
        return torch.load(ckpt_path, map_location=map_location)

    def load_ckpt_to_model(self, model: nn.Module, ckpt_path, model_ema=None):
        sd = self.load_ckpt(ckpt_path)
        if 'base' in sd:
            model.load_state_dict(sd['base'], strict=False)
        if 'lora' in sd:
            model.load_state_dict(sd['lora'], strict=False)
        if 'plugin' in sd:
            model.load_state_dict(sd['plugin'], strict=False)

        if model_ema is not None:
            if 'base' in sd:
                model_ema.load_state_dict(sd['base_ema'])
            if 'lora' in sd:
                model_ema.load_state_dict(sd['lora_ema'])
            if 'plugin' in sd:
                model_ema.load_state_dict(sd['plugin_ema'])

    def save_embedding(self, train_pts, step, replace):
        for k, v in train_pts.items():
            save_path = os.path.join(self.save_dir, f"{k}-{step}.pt")
            save_emb(save_path, v.data, replace=True)
            if replace:
                save_emb(f'{k}.pt', v.data, replace=True)

    def save(self, step, unet, TE, lora_unet, lora_TE, all_plugin_unet, all_plugin_TE, embs, pipe):
        '''

        :param step:
        :param unet:
        :param TE:
        :param lora_unet: [pos, neg]
        :param lora_TE: [pos, neg]
        :param all_plugin_unet:
        :param all_plugin_TE:
        :param emb:
        :param pipe:
        :return:
        '''
        self.save_model_with_lora(unet, lora_unet[0], model_ema=getattr(self, 'ema_unet', None), name='unet', step=step)
        self.save_plugins(unet, all_plugin_unet, name='unet', step=step, model_ema=getattr(self, 'ema_unet', None))

        if TE is not None:
            # exclude_key: embeddings should not save with text-encoder
            self.save_model_with_lora(TE, lora_TE[0], model_ema=getattr(self, 'ema_text_encoder', None),
                                                   name='text_encoder', step=step, exclude_key='emb_ex.')
            self.save_plugins(TE, all_plugin_TE, name='text_encoder', step=step,
                                           model_ema=getattr(self, 'ema_text_encoder', None))

        if lora_unet[1] is not None:
            self.save_model_with_lora(None, lora_unet[1], name='unet-neg', step=step)
            if lora_TE[1] is not None:
                self.save_model_with_lora(None, lora_TE[1], name='text_encoder-neg', step=step)

        self.save_embedding(embs, step, False)

    @classmethod
    def load(cls, pretrained_model):
        raise NotImplementedError(f'{cls} dose not support load()')
