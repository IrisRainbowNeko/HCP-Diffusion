"""
lora.py
====================
    :Name:        lora tools
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import torch
from torch import nn

from hcpdiff.utils.utils import make_mask, low_rank_approximate, isinstance_list
from .plugin import SinglePluginBlock, PluginGroup, BasePluginBlock

from typing import Union, Tuple, Dict, Type

class LoraBlock(SinglePluginBlock):
    wrapable_classes = [nn.Linear, nn.Conv2d]

    def __init__(self, lora_id:int, host:Union[nn.Linear, nn.Conv2d], rank, dropout=0.1, alpha=1.0, bias=False,
                 inplace=True, hook_param=None, alpha_auto_scale=True, **kwargs):
        super().__init__(f'lora_block_{lora_id}', host, hook_param)

        self.mask_range = None
        self.inplace=inplace
        self.bias=bias

        if isinstance(host, nn.Linear):
            self.host_type = 'linear'
            self.layer = self.LinearLayer(host, rank, bias, dropout, self)
        elif isinstance(host, nn.Conv2d):
            self.host_type = 'conv'
            self.layer = self.Conv2dLayer(host, rank, bias, dropout, self)
        else:
            raise NotImplementedError(f'No lora for {type(host)}')
        self.rank = self.layer.rank

        self.register_buffer('alpha', torch.tensor(alpha/self.rank if alpha_auto_scale else alpha))

    def set_mask(self, mask_range):
        self.mask_range = mask_range

    def init_weights(self, svd_init=False):
        host = self.host()
        if svd_init:
            U, V = low_rank_approximate(host.weight, self.rank)
            self.feed_svd(U, V, host.weight)
        else:
            self.layer.lora_down.reset_parameters()
            nn.init.zeros_(self.layer.lora_up.weight)

    def forward(self, fea_in:Tuple[torch.Tensor], fea_out:torch.Tensor):
        if self.mask_range is None:
            return fea_out + self.layer(fea_in[0]) * self.alpha
        else:
            # for DreamArtist-lora
            batch_mask = slice(int(self.mask_range[0]*fea_out.shape[0]), int(self.mask_range[1]*fea_out.shape[0]))
            if self.inplace:
                fea_out[batch_mask, ...] = fea_out[batch_mask, ...] + self.layer(fea_in[0][batch_mask, ...]) * self.alpha
                return fea_out
            else: # colossal-AI dose not support inplace+view
                new_out = fea_out.clone()
                new_out[batch_mask, ...] = fea_out[batch_mask, ...] + self.layer(fea_in[0][batch_mask, ...]) * self.alpha
                return new_out

    def reparameterization_to_host(self, alpha=None, base_alpha=1.0):
        if alpha is None:
            alpha = self.alpha

        host = self.host()
        re_w, re_b = self.layer.get_collapsed_param()
        host.weight = nn.Parameter(
            host.weight.data * base_alpha + alpha * re_w.to(host.weight.device, dtype=host.weight.dtype)
        )

        if self.layer.lora_up.bias is not None:
            if host.bias is None:
                host.bias = nn.Parameter(re_b.to(host.weight.device, dtype=host.weight.dtype))
            else:
                host.bias = nn.Parameter(
                    host.bias.data * base_alpha + alpha * re_b.to(host.weight.device, dtype=host.weight.dtype))

    class LinearLayer(nn.Module):
        def __init__(self, host, rank, bias, dropout, block):
            super().__init__()
            self.rank=rank
            self.bias = bias
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_features * self.rank), 1)
            self.dropout = nn.Dropout(dropout)

        def feed_svd(self, U, V, weight):
            self.lora_up.weight.data = U.to(device=weight.device, dtype=weight.dtype)
            self.lora_down.weight.data = V.to(device=weight.device, dtype=weight.dtype)

        def forward(self, x):
            return self.dropout(self.lora_up(self.lora_down(x)))

        def get_collapsed_param(self) -> Tuple[torch.Tensor, torch.Tensor]:
            pass

    class Conv2dLayer(nn.Module):
        def __init__(self, host, rank, bias, dropout, block):
            super().__init__()
            self.rank = rank
            self.bias = bias
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_channels * self.rank), 1)
            self.dropout = nn.Dropout(dropout)

        def feed_svd(self, U, V, weight):
            self.lora_up.weight.data = U.to(device=weight.device, dtype=weight.dtype)
            self.lora_down.weight.data = V.to(device=weight.device, dtype=weight.dtype)

        def forward(self, x):
            return self.dropout(self.lora_up(self.lora_down(x)))

        def get_collapsed_param(self) -> Tuple[torch.Tensor, torch.Tensor]:
            pass

    @classmethod
    def wrap_layer(cls, lora_id:int, layer: Union[nn.Linear, nn.Conv2d], rank=1, dropout=0.0, alpha=1.0, svd_init=False,
                   bias=False, mask=None, **kwargs):# -> LoraBlock:
        lora_block = cls(lora_id, layer, rank, dropout, alpha, bias=bias, **kwargs)
        lora_block.init_weights(svd_init)
        lora_block.set_mask(mask)
        return lora_block

    @classmethod
    def wrap_model(cls, lora_id:int, model: nn.Module, **kwargs):# -> Dict[str, LoraBlock]:
        return super(LoraBlock, cls).wrap_model(lora_id, model, exclude_key='lora_block_', **kwargs)

    @staticmethod
    def extract_lora_state(model:nn.Module):
        return {k:v for k,v in model.state_dict().items() if 'lora_block_' in k}

    @staticmethod
    def extract_state_without_lora(model:nn.Module):
        return {k:v for k,v in model.state_dict().items() if 'lora_block_' not in k}

    @staticmethod
    def extract_param_without_lora(model:nn.Module):
        return {k:v for k,v in model.named_parameters() if 'lora_block_' not in k}

    @staticmethod
    def extract_trainable_state_without_lora(model:nn.Module):
        trainable_keys = {k for k,v in model.named_parameters() if ('lora_block_' not in k) and v.requires_grad}
        return {k: v for k, v in model.state_dict().items() if k in trainable_keys}

class LoraGroup(PluginGroup):
    def set_mask(self, batch_mask):
        for item in self.plugin_dict.values():
            item.set_mask(batch_mask)

    def collapse_to_host(self, alpha=None, base_alpha=1.0):
        for item in self.plugin_dict.values():
            item.collapse_to_host(alpha, base_alpha)

    def set_inplace(self, inplace):
        for item in self.plugin_dict.values():
            item.inplace = inplace

def split_state(state_dict):
    sd_base, sd_lora={}, {}
    for k, v in state_dict.items():
        if 'lora_block_' in k:
            sd_lora[k]=v
        else:
            sd_base[k]=v
    return sd_base, sd_lora