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
from einops import repeat, rearrange, einsum
from torch import nn

from .lora_base import LoraBlock
from .layers import GroupLinear
import warnings

class LoraLayer(LoraBlock):
    def __init__(self, lora_id:int, host, rank=1, dropout=0.1, alpha=1.0, bias=False, inplace=True, alpha_auto_scale=True, **kwargs):
        super().__init__(lora_id, host, rank, dropout, alpha, bias, inplace, alpha_auto_scale=alpha_auto_scale)

    class LinearLayer(LoraBlock.LinearLayer):
        def __init__(self, host, rank, bias, dropout, block):
            super().__init__(host, rank, bias, dropout, block)
            self.lora_down = nn.Linear(host.in_features, self.rank, bias=False)
            self.lora_up = nn.Linear(self.rank, host.out_features, bias=bias)

        def get_collapsed_param(self):
            w = self.lora_up.weight.data @ self.lora_down.weight.data
            b = self.lora_up.bias.data if self.bias else None
            return w, b

    class Conv2dLayer(LoraBlock.Conv2dLayer):
        def __init__(self, host, rank, bias, dropout, block):
            super().__init__(host, rank, bias, dropout, block)
            self.lora_down = nn.Conv2d(host.in_channels, self.rank, kernel_size=host.kernel_size, stride=host.stride,
                                       padding=host.padding, dilation=host.dilation, groups=host.groups, bias=False)
            self.lora_up = nn.Conv2d(self.rank, host.out_channels, kernel_size=1, stride=1, padding=0,
                                     groups=host.groups, bias=bias)

        def get_collapsed_param(self):
            w = einsum(self.lora_up.weight.data, self.lora_down.weight.data, 'o r ..., r i ... -> o i ...')
            b = self.lora_up.bias.data if self.bias else None
            return w, b

class LoraLayerGroup(LoraBlock):
    def __init__(self, lora_id:int, host, rank=1, dropout=0.1, alpha=1.0, bias=False, inplace=True, rank_groups=1, alpha_auto_scale=True, **kwargs):
        self.rank_groups_raw = rank_groups
        super().__init__(lora_id, host, rank, dropout, alpha, bias, inplace, alpha_auto_scale=alpha_auto_scale)

    def reparameterization_to_host(self, alpha=None, base_alpha=1.0):
        warnings.warn('LoraLayerGroup cannot reparameterization.')
        pass

    class LinearLayer(LoraBlock.LinearLayer):
        def __init__(self, host, rank, bias, dropout, block):
            super().__init__(host, rank, bias, dropout, block)
            self.register_buffer('rank_groups', torch.tensor(block.rank_groups_raw, dtype=torch.int))
            self.lora_down = GroupLinear(host.in_features*self.rank_groups, self.rank, groups=self.rank_groups, bias=False)
            self.lora_up = GroupLinear(self.rank, host.out_features*self.rank_groups, groups=self.rank_groups, bias=bias)

        def feed_svd(self, U, V, weight):
            self.lora_up.weight.data = rearrange(U, 'o (g ri) -> g ri o', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)
            self.lora_down.weight.data = rearrange(V, '(g ri) i -> g i ri', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)

        def forward(self, x):
            x = repeat(x, 'b l c -> b l (g c)', g=self.rank_groups)
            x = rearrange(x, 'b l (g c) -> b g l c', g=self.rank_groups)
            x = self.dropout(self.lora_up(self.lora_down(x)))
            x = torch.prod(x, dim=1, dtype=torch.float16)**(1/self.rank_groups).to(dtype=x.dtype)
            return x

    class Conv2dLayer(LoraBlock.Conv2dLayer):
        def __init__(self, host, rank, bias, dropout, block):
            super().__init__(host, rank, bias, dropout, block)
            self.register_buffer('rank_groups', torch.tensor(block.rank_groups_raw))
            self.lora_down = nn.Conv2d(host.in_channels * self.rank_groups, self.rank, kernel_size=host.kernel_size, stride=host.stride,
                                       padding=host.padding, dilation=host.dilation, groups=self.rank_groups, bias=False)
            self.lora_up = nn.Conv2d(self.rank, host.out_channels * self.rank_groups, kernel_size=1, stride=1,
                                     padding=0, groups=self.rank_groups, bias=bias)

        def feed_svd(self, U, V, weight):
            self.lora_up.weight.data = rearrange(U, 'o (g ri) ... -> (g o) ri ...', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)
            self.lora_down.weight.data = rearrange(V, '(g ri) i ... -> g i ri ...', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)

        def forward(self, x):
            x = self.dropout(self.lora_up(self.lora_down(x)))
            x = torch.prod(rearrange(x, 'b (g c) h w -> b g c h w'), dim=1, dtype=torch.float16) ** (1 / self.rank_groups).to(dtype=x.dtype)
            return x

class LohaLayer(LoraBlock):
    def __init__(self, lora_id:int, host, rank=1, dropout=0.1, alpha=1.0, bias=False, inplace=True, rank_groups=2, alpha_auto_scale=True, **kwargs):
        self.rank_groups_raw = rank_groups
        super().__init__(lora_id, host, rank, dropout, alpha, bias, inplace, hook_param='weight', alpha_auto_scale=alpha_auto_scale)

    def forward(self, host_param: nn.Parameter):
        return host_param + self.layer(host_param) * self.alpha

    class LinearLayer(LoraBlock.LinearLayer):
        def __init__(self, host, rank, bias, dropout, block):
            super().__init__(host, rank, bias, dropout, block)
            self.register_buffer('rank_groups', torch.tensor(block.rank_groups_raw))
            self.W_down = nn.Parameter(torch.empty((block.rank_groups_raw, self.rank//block.rank_groups_raw, host.in_features)))
            self.W_up = nn.Parameter(torch.empty((block.rank_groups_raw, host.out_features, self.rank//block.rank_groups_raw)))

        def forward(self, x):
            return torch.prod(self.W_up @ self.W_down, dim=0)

        def feed_svd(self, U, V, weight):
            self.W_up.data = rearrange(U, 'o (g ri) -> g ri o', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)
            self.W_down.data = rearrange(V, '(g ri) i -> g i ri', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)

        def get_collapsed_param(self):
            w = torch.prod(self.W_up.data @ self.W_down.data, dim=0)
            b = None
            return w, b

    class Conv2dLayer(LoraBlock.Conv2dLayer):
        def __init__(self, host, rank, bias, dropout, block):
            super().__init__(host, rank, bias, dropout, block)
            self.W_down = nn.Parameter( torch.empty((block.rank_groups_raw, self.rank // block.rank_groups_raw,
                                                     host.in_features, *host.kernel_size)))
            self.W_up = nn.Parameter(torch.empty((block.rank_groups_raw, host.out_features,
                                                    self.rank//block.rank_groups_raw, *([1]*len(host.kernel_size)))))

        def forward(self, x):
            return torch.prod(einsum(self.W_up, self.W_down, 'g o r ..., g r i ... -> g o i ...'), dim=0)

        def feed_svd(self, U, V, weight):
            self.W_up.data = rearrange(U, 'o (g ri) ... -> (g o) ri ...', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)
            self.W_down.data = rearrange(V, '(g ri) i ... -> g i ri ...', g=self.rank_groups).to(device=weight.device, dtype=weight.dtype)

        def get_collapsed_param(self):
            w = torch.prod(einsum(self.W_up.data, self.W_down.data, 'g o r ..., g r i ... -> g o i ...'), dim=0)
            b = None
            return w, b

layer_map={
    'lora': LoraLayer,
    'loha_group': LoraLayerGroup,
    'loha': LohaLayer,
}
