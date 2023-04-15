"""
lora_layers.py
====================
    :Name:        lora layers
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     09/04/2023
    :Licence:     Apache-2.0
"""

from typing import Union

import torch
from einops import repeat, rearrange
from torch import nn

from .lora_base import LoraBlock
from .layers import GroupLinear

class LoraLayer(LoraBlock):
    def __init__(self, host, rank, dropout=0.1, bias=False):
        super().__init__(host, rank, dropout, bias)

    class LinearLayer(LoraBlock.LinearLayer):
        def __init__(self, host, rank, bias):
            super().__init__(host, rank, bias)
            self.lora_down = nn.Linear(host.in_features, self.rank, bias=False)
            self.lora_up = nn.Linear(self.rank, host.out_features, bias=self.bias)

        def get_collapsed_param(self):
            w = self.lora_up.weight.data @ self.lora_up.weight.data
            b = self.lora_up.bias.data if self.bias else None
            return w, b

    class Conv2dLayer(LoraBlock.Conv2dLayer):
        def __init__(self, host, rank, bias):
            super().__init__(host, rank, bias)
            self.lora_down = nn.Conv2d(host.in_channels, self.rank, kernel_size=host.kernel_size, stride=host.stride,
                                       padding=host.padding, dilation=host.dilation, groups=host.groups, bias=False)
            self.lora_up = nn.Conv2d(self.rank, host.out_channels, kernel_size=1, stride=1, padding=0,
                                     groups=host.groups, bias=self.bias)

        def get_collapsed_param(self):
            w = self.lora_up.weight.data.flatten(1) @ self.lora_up.weight.data.flatten(1)
            b = self.lora_up.bias.data if self.bias else None
            return w, b

class LoraLayerGroup(LoraBlock):
    def __init__(self, host, rank, dropout=0.1, bias=False, rank_groups=1):
        self.rank_groups_raw = rank_groups
        super().__init__(host, rank, dropout, bias)

    def collapse_to_host(self, alpha=None, base_alpha=1.0):
        raise NotImplementedError('LoraLayerGroup not support reparameterization.')

    class LinearLayer(LoraBlock.LinearLayer):
        def __init__(self, host, rank, bias):
            super().__init__(host, rank, bias)
            self.register_buffer('rank_groups', torch.tensor(self.rank_groups_raw, dtype=torch.int))
            self.lora_down = GroupLinear(host.in_features*self.rank_groups, self.rank, groups=self.rank_groups, bias=False)
            self.lora_up = GroupLinear(self.rank, host.out_features*self.rank_groups, groups=self.rank_groups, bias=self.bias)

        def feed_svd(self, U, V, weight):
            self.lora_up.weight.data = rearrange(U, 'o (g ri) -> g ri o', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)
            self.lora_down.weight.data = rearrange(V, '(g ri) i -> g i ri', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)

        def lora_forward(self, x):
            x = repeat(x, 'b l c -> b l (g c)', g=self.rank_groups)
            x = rearrange(x, 'b l (g c) -> b g l c', g=self.rank_groups)
            x = self.dropout(self.lora_up(self.lora_down(x)))
            x = torch.prod(x, dim=1, dtype=torch.float16)**(1/self.rank_groups).to(dtype=x.dtype)
            return x

    class Conv2dLayer(LoraBlock.Conv2dLayer):
        def __init__(self, host, rank, bias):
            super().__init__(host, rank, bias)
            self.register_buffer('rank_groups', torch.tensor(self.rank_groups_raw))
            self.lora_down = nn.Conv2d(host.in_channels * self.rank_groups, self.rank, kernel_size=host.kernel_size, stride=host.stride,
                                       padding=host.padding, dilation=host.dilation, groups=self.rank_groups, bias=False)
            self.lora_up = nn.Conv2d(self.rank, host.out_channels * self.rank_groups, kernel_size=1, stride=1,
                                     padding=0, groups=self.rank_groups, bias=self.bias)

        def feed_svd(self, U, V, weight):
            self.lora_up.weight.data = rearrange(U, 'o (g ri) -> (g o) ri', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)
            self.lora_down.weight.data = V.to(device=weight.device, dtype=weight.dtype)

        def lora_forward(self, x):
            x = self.dropout(self.lora_up(self.lora_down(x)))
            x = torch.prod(rearrange(x, 'b (g c) h w -> b g c h w'), dim=1, dtype=torch.float16) ** (1 / self.rank_groups).to(dtype=x.dtype)
            return x

layer_map={
    'lora': LoraLayer,
    'loha_group': LoraLayerGroup,
}
