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
from torch.nn import functional as F

from hcpdiff.utils.utils import make_mask, low_rank_approximate, isinstance_list
from .plugin import PatchPluginBlock, PluginGroup, PatchPluginContainer

from typing import Union, Tuple, Dict, Type

class LoraPatchContainer(PatchPluginContainer):
    def forward(self, x, *args, **kwargs):
        weight_ = None
        bias_ = None
        for name in self.plugin_names:
            if weight_ is None:
                weight_ = self[name].get_weight()
            else:
                weight_ = weight_ + self[name].get_weight()

            if bias_ is None:
                bias_ = self[name].get_bias()
            else:
                bias_ = bias_ + self[name].get_bias()

        return self[name].post_forward(x, self._host.weight, weight_, self._host.bias, bias_)

class LoraBlock(PatchPluginBlock):
    container_cls = LoraPatchContainer
    wrapable_classes = (nn.Linear, nn.Conv2d)

    def __init__(self, lora_id:int, host:Union[nn.Linear, nn.Conv2d], rank, dropout=0.1, alpha=1.0, bias=False,
                 alpha_auto_scale=True, parent_block=None, host_name=None, **kwargs):
        super().__init__(f'lora_block_{lora_id}', host, parent_block=parent_block, host_name=host_name)

        self.bias=bias

        host = self.host()
        if isinstance(host, nn.Linear):
            self.host_type = 'linear'
            self.layer = self.LinearLayer(host, rank, bias, self)
        elif isinstance(host, nn.Conv2d):
            self.host_type = 'conv'
            self.layer = self.Conv2dLayer(host, rank, bias, self)
        else:
            raise NotImplementedError(f'No lora for {type(host)}')
        self.dropout = nn.Dropout(dropout)

        self.rank = self.layer.rank
        self.register_buffer('alpha', torch.tensor(alpha/self.rank if alpha_auto_scale else alpha))

    def get_weight(self):
        return self.layer.get_weight() * self.alpha

    def get_bias(self):
        bias = self.layer.get_bias()
        return bias * self.alpha if bias is not None else None

    def post_forward(self, x, host_weight, weight, host_bias, bias=None):
        if host_bias is not None:
            if bias is None:
                bias = host_bias
            else:
                bias = host_bias + bias
        return self.dropout(self.layer(x, host_weight+weight, bias))

    def init_weights(self, svd_init=False):
        if svd_init:
            host = self.host()
            U, V = low_rank_approximate(host.weight, self.rank)
            self.layer.feed_svd(U, V, host.weight)
        else:
            self.layer.reset_parameters()

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
        def __init__(self, host, rank, bias, block):
            super().__init__()
            self.rank=rank
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_features * self.rank), 1)

        def feed_svd(self, U, V, weight):
            self.lora_up.weight.data = U.to(device=weight.device, dtype=weight.dtype)
            self.lora_down.weight.data = V.to(device=weight.device, dtype=weight.dtype)

        def get_weight(self) -> torch.Tensor:
            pass

        def get_bias(self) -> torch.Tensor:
            pass

        def forward(self, x, weight, bias=None):
            pass

        def get_collapsed_param(self) -> Tuple[torch.Tensor, torch.Tensor]:
            pass

    class Conv2dLayer(nn.Module):
        def __init__(self, host, rank, bias, block):
            super().__init__()
            self.rank = rank
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_channels * self.rank), 1)

        def feed_svd(self, U, V, weight):
            self.lora_up.weight.data = U.to(device=weight.device, dtype=weight.dtype)
            self.lora_down.weight.data = V.to(device=weight.device, dtype=weight.dtype)

        def get_weight(self) -> torch.Tensor:
            pass

        def get_bias(self) -> torch.Tensor:
            pass

        def forward(self, x, weight, bias=None):
            pass

        def get_collapsed_param(self) -> Tuple[torch.Tensor, torch.Tensor]:
            pass

    @classmethod
    def wrap_layer(cls, lora_id:int, layer: Union[nn.Linear, nn.Conv2d], rank=1, dropout=0.0, alpha=1.0, svd_init=False,
                   bias=False, mask=None, **kwargs):# -> LoraBlock:
        lora_block = cls(lora_id, layer, rank, dropout, alpha, bias=bias, **kwargs)
        lora_block.init_weights(svd_init)
        return lora_block

    @classmethod
    def wrap_model(cls, lora_id:int, model: nn.Module, **kwargs):# -> Dict[str, LoraBlock]:
        return super(LoraBlock, cls).wrap_model(lora_id, model, exclude_classes=(LoraBlock,), **kwargs)

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