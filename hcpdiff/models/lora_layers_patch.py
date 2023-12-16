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
from einops import einsum, rearrange
from torch import nn
from torch.nn import functional as F

from .lora_base_patch import LoraBlock, PatchPluginContainer
from .layers import GroupLinear
import math
from typing import Union, List

class LoraLayer(LoraBlock):
    def __init__(self, lora_id: int, host, rank=1, dropout=0.1, alpha=1.0, bias=False, alpha_auto_scale=True, **kwargs):
        super().__init__(lora_id, host, rank, dropout, alpha=alpha, bias=bias, alpha_auto_scale=alpha_auto_scale, **kwargs)

    class LinearLayer(LoraBlock.LinearLayer):
        def __init__(self, host:nn.Linear, rank, bias, block):
            super().__init__(host, rank, bias, block)
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_features * self.rank), 1)

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
            # make it faster
            x_shape = x.shape
            if bias is None:
                return torch.mm(x.view(-1, x_shape[-1]), weight.transpose(0, 1)).view(*x_shape[:-1], -1)
            else:
                return torch.mm(x.view(-1, x_shape[-1]), weight.transpose(0, 1)).view(*x_shape[:-1], -1) + bias
            #return F.linear(x, weight, bias) # linear is slow

        def get_collapsed_param(self):
            w = self.W_up.data@self.W_down.data
            b = self.bias.data if self.bias else None
            return w, b

    class Conv2dLayer(LoraBlock.Conv2dLayer):
        def __init__(self, host: nn.Conv2d, rank, bias, block):
            super().__init__(host, rank, bias, block)
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_channels * self.rank), 1)

            self.W_down = nn.Parameter(torch.empty(self.rank, host.in_channels, *host.kernel_size))
            self.W_up = nn.Parameter(torch.empty(host.out_channels, self.rank, 1, 1))
            if bias:
                self.bias = nn.Parameter(torch.empty(host.out_channels))
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

class DAPPPatchContainer(PatchPluginContainer):
    def forward(self, x, *args, **kwargs):
        weight_p = None
        weight_n = None
        bias_p = None
        bias_n = None
        for name in self.plugin_names:
            if self[name].branch=='p':
                if weight_p is None:
                    weight_p = self[name].get_weight()
                else:
                    weight_p = weight_p + self[name].get_weight()

                if bias_p is None:
                    bias_p = self[name].get_bias()
                else:
                    bias_p = bias_p+self[name].get_bias()
            elif self[name].branch=='n':
                if weight_n is None:
                    weight_n = self[name].get_weight()
                else:
                    weight_n = weight_n + self[name].get_weight()

                if bias_n is None:
                    bias_n = self[name].get_bias()
                else:
                    bias_n = bias_n+self[name].get_bias()

        B = x.shape[0]//2
        x_p = self[name].post_forward(x[B:], self._host.weight, weight_p, self._host.bias, bias_p)
        x_n = self[name].post_forward(x[:B], self._host.weight, weight_n, self._host.bias, bias_n)
        return torch.cat([x_n, x_p], dim=0)

class DAPPLayer(LoraBlock):
    container_cls = DAPPPatchContainer
    def __init__(self, lora_id: int, host, rank=1, dropout=0.1, alpha=1.0, bias=False, alpha_auto_scale=True, branch='p', **kwargs):
        super().__init__(lora_id, host, rank, dropout, alpha=alpha, bias=bias, alpha_auto_scale=alpha_auto_scale, **kwargs)
        self.branch = branch

    class LinearLayer(LoraBlock.LinearLayer):
        def __init__(self, host:nn.Linear, rank, bias, block):
            super().__init__(host, rank, bias, block)
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_features * self.rank), 1)

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
            # make it faster
            x_shape = x.shape
            if bias is None:
                return torch.mm(x.view(-1, x_shape[-1]), weight.transpose(0, 1)).view(*x_shape[:-1], -1)
            else:
                return torch.mm(x.view(-1, x_shape[-1]), weight.transpose(0, 1)).view(*x_shape[:-1], -1) + bias
            #return F.linear(x, weight, bias) # linear is slow

        def get_collapsed_param(self):
            w = self.W_up.data@self.W_down.data
            b = self.bias.data if self.bias else None
            return w, b

    class Conv2dLayer(LoraBlock.Conv2dLayer):
        def __init__(self, host: nn.Conv2d, rank, bias, block):
            super().__init__(host, rank, bias, block)
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_channels * self.rank), 1)

            self.W_down = nn.Parameter(torch.empty(self.rank, host.in_channels, *host.kernel_size))
            self.W_up = nn.Parameter(torch.empty(host.out_channels, self.rank, 1, 1))
            if bias:
                self.bias = nn.Parameter(torch.empty(host.out_channels))
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
    'dapp':DAPPLayer,
}
