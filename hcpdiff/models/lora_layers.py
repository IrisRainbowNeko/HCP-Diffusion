"""
lora.py
====================
    :Name:        lora tools
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     MIT
"""

from typing import Union

import torch
from einops import repeat, rearrange
from torch import nn

from hcpdiff.utils.utils import low_rank_approximate
from .layers import GroupLinear


class LoraLayerBase(nn.Module):
    def __init__(self, host: Union[nn.Linear, nn.Conv2d], rank, dropout=0.1, bias=False):
        super().__init__()
        self.host = host
        self.rank = rank
        self.dropout = nn.Dropout(dropout)
        self.host_type = None
        self.bias = bias

        if isinstance(self.rank, float):
            self.rank = max(round(host.out_features * self.rank), 1)

        self.build_layers()

    def build_layers(self):
        pass

    def feed_sdv(self, U, V, weight):
        self.lora_up.weight.data = U.to(device=weight.device, dtype=weight.dtype)
        self.lora_down.weight.data = V.to(device=weight.device, dtype=weight.dtype)

    def init_weights(self, svd_init=False):
        host = self.host()
        if svd_init:
            U, V = low_rank_approximate(host.weight, self.rank)
            self.feed_sdv(U, V, host.weight)
        else:
            self.lora_down.reset_parameters()
            nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        x = self.dropout(self.lora_up(self.lora_down(x)))
        return x

    def get_collapsed_param(self):
        w = collapse_lora_weight(self.lora_up.weight.data, self.lora_up.weight.data, self.host_type)
        b = self.lora_up.bias.data
        return w, b

    def collapse_to_host(self, alpha=None, base_alpha=1.0):
        if alpha is None:
            alpha = self.scale

        host = self.host()
        re_w, re_b = self.get_collapsed_param()
        host.weight = nn.Parameter(
            host.weight.data * base_alpha + alpha * re_w.to(host.weight.device, dtype=host.weight.dtype)
        )

        if self.lora_up.bias is not None:
            if host.bias is None:
                host.bias = nn.Parameter(re_b.to(host.weight.device, dtype=host.weight.dtype))
            else:
                host.bias = nn.Parameter(
                    host.bias.data * base_alpha + alpha * re_b.to(host.weight.device, dtype=host.weight.dtype))


class LoraLayerLinear(LoraLayerBase):
    def __init__(self, host, rank, dropout=0.1, bias=False):
        super().__init__(host, rank, dropout, bias)
        self.host_type = 'linear'

    def build_layers(self):
        host = self.host()
        self.lora_down = nn.Linear(host.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, host.out_features, bias=self.bias)


class LoraLayerConv2d(LoraLayerBase):
    def __init__(self, host, rank, dropout=0.1, bias=False):
        super().__init__(host, rank, dropout, bias)
        self.host_type = 'conv'

    def build_layers(self):
        host = self.host()
        self.lora_down = nn.Conv2d(host.in_channels, self.rank, kernel_size=host.kernel_size, stride=host.stride,
                                   padding=host.padding, dilation=host.dilation, groups=host.groups, bias=False)
        self.lora_up = nn.Conv2d(self.rank, host.out_channels, kernel_size=1, stride=1, padding=0, groups=host.groups, bias=self.bias)

class LoraLayerLinearGroup(LoraLayerLinear):
    def __init__(self, host, rank, dropout=0.1, bias=False, rank_groups=1):
        self.rank_groups_raw = rank_groups
        super().__init__(host, rank, dropout, bias)

    def build_layers(self):
        host = self.host()
        self.register_buffer('rank_groups', torch.tensor(self.rank_groups_raw, dtype=torch.int))
        self.lora_down = GroupLinear(host.in_features*self.rank_groups, self.rank, groups=self.rank_groups, bias=False)
        self.lora_up = GroupLinear(self.rank, host.out_features*self.rank_groups, groups=self.rank_groups, bias=self.bias)

    def feed_sdv(self, U, V, weight):
        self.lora_up.weight.data = rearrange(U, 'o (g ri) -> g ri o', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)
        self.lora_down.weight.data = rearrange(V, '(g ri) i -> g i ri', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)

    def forward(self, x):
        x = repeat(x, 'b l c -> b l (g c)', g=self.rank_groups)
        x = rearrange(x, 'b l (g c) -> b g l c', g=self.rank_groups)
        x = self.dropout(self.lora_up(self.lora_down(x)))
        x = torch.prod(x, dim=1, dtype=torch.float16)
        return x

    def get_collapsed_param(self):
        raise NotImplementedError('LoraLayerLinearGroup not support reparameterization.')


class LoraLayerConv2dGroup(LoraLayerLinear):
    def __init__(self, host, rank, dropout=0.1, bias=False, rank_groups=1):
        self.rank_groups_raw = rank_groups
        super().__init__(host, rank, dropout, bias)

    def build_layers(self):
        host = self.host()
        self.register_buffer('rank_groups', torch.tensor(self.rank_groups_raw))
        self.lora_down = nn.Conv2d(host.in_channels*self.rank_groups, self.rank, kernel_size=host.kernel_size, stride=host.stride,
                                   padding=host.padding, dilation=host.dilation, groups=self.rank_groups, bias=False)
        self.lora_up = nn.Conv2d(self.rank, host.out_channels * self.rank_groups, kernel_size=1, stride=1, padding=0,
                                 groups=self.rank_groups, bias=self.bias)

    def feed_sdv(self, U, V, weight):
        self.lora_up.weight.data = rearrange(U, 'o (g ri) -> (g o) ri', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)
        self.lora_down.weight.data = V.to(device=weight.device, dtype=weight.dtype)

    def forward(self, x):
        x = self.dropout(self.lora_up(self.lora_down(x)))
        x = torch.prod(rearrange(x, 'b (g c) h w -> b g c h w'), dim=1)
        return x

    def get_collapsed_param(self):
        raise NotImplementedError('LoraLayerConv2dGroup not support reparameterization.')


def collapse_lora_weight(lora_up, lora_down, host_type):
    if host_type == 'linear':
        return lora_up @ lora_down
    elif host_type == 'conv':
        return lora_up.flatten(1) @ lora_down.flatten(1)

def build_layer(host, host_type, rank, dropout=0.1, bias=False, rank_groups=1):
    if rank_groups > 1:
        if host_type == 'linear':
            return LoraLayerLinearGroup(host, rank, dropout, bias, rank_groups)
        elif host_type == 'conv':
            return LoraLayerLinearGroup(host, rank, dropout, bias, rank_groups)
    else:
        if host_type == 'linear':
            return LoraLayerLinear(host, rank, dropout, bias)
        elif host_type == 'conv':
            return LoraLayerConv2d(host, rank, dropout, bias)
