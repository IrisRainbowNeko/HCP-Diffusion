"""
lora_layers.py
====================
    :Name:        lora layers
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     09/04/2023
    :Licence:     Apache-2.0
"""

import torch
from einops import einsum
from torch import nn
from torch.nn import functional as F

from .lora_base_patch import LoraBlock
import math

class LoraLayer(LoraBlock):
    def __init__(self, lora_id: int, host, rank=1, dropout=0.1, alpha=1.0, bias=False, alpha_auto_scale=True, **kwargs):
        super().__init__(lora_id, host, rank, dropout, alpha=alpha, bias=bias, alpha_auto_scale=alpha_auto_scale, **kwargs)

    class LinearLayer(LoraBlock.LinearLayer):
        def __init__(self, host:nn.Linear, rank, bias, block):
            super().__init__(host, rank, bias, block)
            self.W_down = nn.Parameter(torch.empty(self.rank, host.in_features))
            self.W_up = nn.Parameter(torch.empty(host.out_features, self.rank))
            if bias:
                self.bias = nn.Parameter(torch.empty(host.out_features))
            else:
                self.register_parameter('bias', None)

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
            nn.init.zeros_(self.W_up)
            if self.bias:
                nn.init.zeros_(self.bias)

        def get_weight(self):
            return torch.mm(self.W_up, self.W_down)

        def get_bias(self):
            return self.bias

        def forward(self, x, weight, bias=None):
            return F.linear(x, weight, bias)

        def get_collapsed_param(self):
            w = self.W_up.data@self.W_down.data
            b = self.bias.data if self.bias else None
            return w, b

    class Conv2dLayer(LoraBlock.Conv2dLayer):
        def __init__(self, host: nn.Conv2d, rank, bias, block):
            super().__init__(host, rank, bias, block)
            self.W_down = nn.Parameter(torch.empty(self.rank, host.in_features, *host.kernel_size))
            self.W_up = nn.Parameter(torch.empty(host.out_features, self.rank, 1, 1))
            if bias:
                self.bias = nn.Parameter(torch.empty(host.out_features))
            else:
                self.register_parameter('bias', None)

            self.stride = host.stride
            self.padding = host.padding
            self.dilation = host.dilation
            self.groups = host.groups

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
            nn.init.zeros_(self.W_up)
            if self.bias:
                nn.init.zeros_(self.bias)

        def get_weight(self):
            return einsum(self.W_up, self.W_down, 'o r ..., r i ... -> o i ...')

        def get_bias(self):
            return self.bias if self.bias else None

        def forward(self, x, weight, bias=None):
            return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        def get_collapsed_param(self):
            w = einsum(self.W_up.data, self.W_down.data, 'o r ..., r i ... -> o i ...')
            b = self.bias.data if self.bias else None
            return w, b

lora_layer_map = {
    'lora':LoraLayer,
}
