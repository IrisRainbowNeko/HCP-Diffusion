"""
lora.py
====================
    :Name:        lora tools
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     MIT
"""

import torch
from torch import nn
import math
from einops import repeat

from utils.utils import low_rank_approximate, make_mask
from .plugin import SinglePluginBlock, PluginGroup

from typing import Dict, Union, Tuple

class LoraBlock(SinglePluginBlock):
    def __init__(self, host:Union[nn.Linear, nn.Conv2d], rank, dropout=0.1, scale=1.0, bias=False, inplace=True):
        super().__init__(host)
        if hasattr(host, 'lora_block'):
            self.id = len(host.lora_block)
            host.lora_block.append(self)
        else:
            self.id = 0
            host.lora_block=nn.ModuleList([self])

        self.rank=rank
        self.dropout=dropout
        self.mask_range = None
        self.inplace=inplace

        self.dropout = nn.Dropout(dropout)
        if isinstance(host, nn.Linear):
            self.host_type='linear'
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_features*self.rank), 1)
            self.lora_down = nn.Linear(host.in_features, self.rank, bias=False)
            self.lora_up = nn.Linear(self.rank, host.out_features, bias=bias)
        elif isinstance(host, nn.Conv2d):
            self.host_type = 'conv'
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_channels*self.rank), 1)
            self.lora_down = nn.Conv2d(host.in_channels, self.rank, kernel_size=host.kernel_size, stride=host.stride,
                                        padding=host.padding, dilation=host.dilation, groups=host.groups, bias=False)
            self.lora_up = nn.Conv2d(self.rank, host.out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        else:
            raise NotImplementedError('lora support only Linear and Conv2d now.')
        self.register_buffer('scale', torch.tensor(1.0 if scale == 0 else scale / self.rank))

    def set_mask(self, mask_range):
        self.mask_range = mask_range

    def init_weights(self, svd_init=False):
        host=self.host()
        if svd_init:
            U, V = low_rank_approximate(host.weight, self.rank)
            self.lora_up.weight.data = U.to(device=host.weight.device, dtype=host.weight.dtype)
            self.lora_down.weight.data = V.to(device=host.weight.device, dtype=host.weight.dtype)
        else:
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

    def forward(self, fea_in:Tuple[torch.Tensor], fea_out:torch.Tensor):
        if self.mask_range is None:
            return fea_out + self.dropout(self.lora_up(self.lora_down(fea_in[0]))) * self.scale
        else:
            # for DreamArtist-lora
            batch_mask = make_mask(self.mask_range[0], self.mask_range[1], fea_out.shape[0])
            if self.inplace:
                fea_out[batch_mask, ...] = fea_out[batch_mask, ...] \
                                       + self.dropout(self.lora_up(self.lora_down(fea_in[0][batch_mask, ...]))) * self.scale
                return fea_out
            else: # colossal-AI dose not support inplace+view
                new_out = fea_out.clone()
                new_out[batch_mask, ...] = fea_out[batch_mask, ...] \
                                    + self.dropout(self.lora_up(self.lora_down(fea_in[0][batch_mask, ...]))) * self.scale
                return new_out

    def remove(self):
        super().remove()
        host = self.host()
        for i in range(len(host.lora_block)):
            if host.lora_block[i] == self:
                del host.lora_block[i]
                break
        if len(host.lora_block)==0:
            del host.lora_block

    def collapse_to_host(self, alpha=None, base_alpha=1.0):
        if alpha is None:
            alpha=self.scale

        host = self.host()
        host.weight = nn.Parameter(
            host.weight.data*base_alpha +
            alpha * collapse_lora_weight(self.lora_up.weight.data, self.lora_down.weight.data, self.host_type)
            .to(host.weight.device, dtype=host.weight.dtype)
        )

        if self.lora_up.bias is not None:
            if host.bias is None:
                host.bias = nn.Parameter(self.lora_up.bias.data.to(host.weight.device, dtype=host.weight.dtype))
            else:
                host.bias = nn.Parameter(host.bias.data*base_alpha +
                                alpha*self.lora_up.bias.data.to(host.weight.device, dtype=host.weight.dtype))

    @classmethod
    def warp_layer(cls, layer: Union[nn.Linear, nn.Conv2d], rank, dropout=0.1, scale=1.0, svd_init=False, bias=False, mask=None):# -> LoraBlock:
        lora_block = cls(layer, rank, dropout, scale, bias=bias)
        lora_block.init_weights(svd_init)
        lora_block.set_mask(mask)
        return lora_block

    @classmethod
    def warp_model(cls, model: nn.Module, rank, dropout=0.0, scale=1.0, svd_init=False, bias=False, mask=None, **kwargs):# -> Dict[str, LoraBlock]:
        lora_block_dict = {}
        if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
            lora_block_dict['lora_block'] = cls.warp_layer(model, rank, dropout, scale, svd_init, bias=bias, mask=mask)
        else:
            # there maybe multiple lora block, avoid insert lora into lora_block
            named_modules = {name: layer for name, layer in model.named_modules() if 'lora_block' not in name}
            for name, layer in named_modules.items():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    lora_block_dict[f'{name}.lora_block'] = cls.warp_layer(layer, rank, dropout, scale, svd_init, bias=bias, mask=mask)
        return lora_block_dict

    @staticmethod
    def extract_lora_state(model:nn.Module):
        return {k:v for k,v in model.state_dict().items() if 'lora_block.' in k}

    @staticmethod
    def extract_state_without_lora(model:nn.Module):
        return {k:v for k,v in model.state_dict().items() if 'lora_block.' not in k}

    @staticmethod
    def extract_param_without_lora(model:nn.Module):
        return {k:v for k,v in model.named_parameters() if 'lora_block.' not in k}

    @staticmethod
    def extract_trainable_state_without_lora(model:nn.Module):
        trainable_keys = {k for k,v in model.named_parameters() if ('lora_block.' not in k) and v.requires_grad}
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

#TODO: Inter DreamArtist
class InterDALoraBlock(SinglePluginBlock):
    def __init__(self, host: nn.Module, lora_pos: LoraBlock, lora_neg: LoraBlock):
        super().__init__(host)
        self.lora_pos = lora_pos
        self.lora_neg = lora_neg

    def set_cfg_context(self, cfg_context):
        self.cfg_context = cfg_context

    def forward(self, fea_in:Tuple[torch.Tensor], fea_out:torch.Tensor):
        if self.mask_range is None:
            return fea_out + self.dropout(self.lora_up(self.lora_down(fea_in[0]))) * self.scale
        else:
            # for DreamArtist-lora
            pos_mask = make_mask(0.5, 1, fea_out.shape[0])
            neg_mask = make_mask(0, 0.5, fea_out.shape[0])
            fea_out[pos_mask, ...] = self.lora_pos(fea_out[pos_mask, ...])
            fea_out[neg_mask, ...] = self.lora_neg(fea_out[neg_mask, ...])
            fea_out = repeat(self.cfg_context.post(fea_out), 'b ... -> (pn b) ...', pn=2)
            return fea_out

    @classmethod
    def modify_layer(cls, layer:nn.Module):
        block_dict={}
        for block in layer.lora_block:
            if block.mask_range[1]==0.5:
                block_dict['lora_neg']=block
            elif block.mask_range[0]==0.5:
                block_dict['lora_pos']=block
        if len(block_dict)!=2:
            return
        return cls(layer, **block_dict)

    @classmethod
    def modify_model(cls, model: nn.Module):
        lora_block_dict = {}
        if hasattr(model, 'lora_block'):
            lora_block_dict['lora_block'] = cls.modify_layer(model)
        else:
            # there maybe multiple lora block, avoid insert lora into lora_block
            named_modules = {name: layer for name, layer in model.named_modules() if 'lora_block' in name}
            for name, layer in named_modules.items():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    lora_block_dict[f'{name}.lora_block'] = cls.warp_layer(layer, rank, dropout, scale, svd_init, bias=bias, mask=mask)
        return lora_block_dict

def collapse_lora_weight(lora_up, lora_down, host_type):
    if host_type == 'linear':
        return lora_up @ lora_down
    elif host_type == 'conv':
        return lora_up.flatten(1) @ lora_down.flatten(1)

def split_state(state_dict):
    sd_base, sd_lora={}, {}
    for k, v in state_dict.items():
        if 'lora_block.' in k:
            sd_lora[k]=v
        else:
            sd_base[k]=v
    return sd_base, sd_lora