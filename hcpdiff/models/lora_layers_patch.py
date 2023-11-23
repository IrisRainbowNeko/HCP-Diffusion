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

from .lora_base_patch import LoraBlock
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

class DAPPLayer(LoraBlock):
    def __init__(self, lora_id: int, host, rank=1, dropout=0.1, alpha=1.0, bias=False, alpha_auto_scale=True, **kwargs):
        super().__init__(lora_id, host, rank, dropout, alpha=alpha, bias=bias, alpha_auto_scale=alpha_auto_scale, **kwargs)

    def post_forward(self, x, host_weight, weight, host_bias, bias=None):
        weight = host_weight.unsqueeze(0)+weight
        if host_bias is not None:
            if bias is None:
                bias = host_bias.view(1, 1, -1).repeat(2, 1, 1)
            else:
                bias = host_bias.view(1,1,-1)+bias
        return self.dropout(self.layer(x, weight, bias))

    class LinearLayer(LoraBlock.LinearLayer):
        def __init__(self, host:nn.Linear, rank:Union[int, List[int]], bias, block):
            super().__init__(host, rank, bias, block)

            # build rank mask
            if isinstance(self.rank, int):
                mask = torch.ones(2, self.rank)
            else:
                if isinstance(self.rank[0], float):
                    self.rank[0] = max(round(host.out_features*self.rank[0]), 1)
                if isinstance(self.rank[1], float):
                    self.rank[1] = max(round(host.out_features*self.rank[1]), 1)

                # mask unused parameters
                if self.rank[0]>self.rank[1]:
                    mask = torch.ones(2, self.rank[0])
                    mask[1,self.rank[1]:] = 0.
                    self.rank = self.rank[0]
                else:
                    mask = torch.ones(2, self.rank[1])
                    mask[0, self.rank[1]:] = 0.
                    self.rank = self.rank[1]
            self.register_buffer('mask_down', mask.unsqueeze(2))
            self.register_buffer('mask_up', mask.unsqueeze(1))

            self.W_down = nn.Parameter(torch.empty(2, self.rank, host.in_features))
            self.W_up = nn.Parameter(torch.empty(2, host.out_features, self.rank))
            if bias:
                self.bias = nn.Parameter(torch.empty(2, 1, host.out_features))
            else:
                self.register_parameter('bias', None)

        def reset_parameters(self):
            GroupLinear.kaiming_uniform_group(self.W_down, a=math.sqrt(5))
            self.W_down.data = self.W_down.data*self.mask_down
            nn.init.zeros_(self.W_up)
            if self.bias:
                nn.init.zeros_(self.bias)

        def get_weight(self):
            return torch.bmm(self.W_up*self.mask_up, self.W_down*self.mask_down)

        def get_bias(self):
            return self.bias

        def forward(self, x, weight, bias=None):
            B = x.shape[0]//2
            if len(x.shape)==3:
                x = rearrange(x, '(g b) l c -> g (b l) c', g=2)
                if bias is None:
                    out = torch.bmm(x, weight.transpose(1, 2))
                else:
                    out = torch.bmm(x, weight.transpose(1, 2))+bias
                out = rearrange(out, 'g (b l) c -> (g b) l c', b=B)
            else:
                x = rearrange(x, '(g b) c -> g b c', g=2)
                if bias is None:
                    out = torch.bmm(x, weight.transpose(1, 2))
                else:
                    out = torch.bmm(x, weight.transpose(1, 2))+bias
                out = rearrange(out, 'g b c -> (g b) c', b=B)
            return out

        def get_collapsed_param(self):
            w = self.W_up.data@self.W_down.data
            b = self.bias.data if self.bias else None
            return w, b

    class Conv2dLayer(LoraBlock.Conv2dLayer):
        def __init__(self, host: nn.Conv2d, rank, bias, block):
            super().__init__(host, rank, bias, block)

            # build rank mask
            if isinstance(self.rank, int):
                mask = torch.ones(2, self.rank)
            else:
                if isinstance(self.rank[0], float):
                    self.rank[0] = max(round(host.out_channels*self.rank[0]), 1)
                if isinstance(self.rank[1], float):
                    self.rank[1] = max(round(host.out_channels*self.rank[1]), 1)

                # mask unused parameters
                if self.rank[0]>self.rank[1]:
                    mask = torch.ones(2, self.rank[0])
                    mask[1,self.rank[1]:] = 0.
                    self.rank = self.rank[0]
                else:
                    mask = torch.ones(2, self.rank[1])
                    mask[0, self.rank[1]:] = 0.
                    self.rank = self.rank[1]
            self.register_buffer('mask_down', mask.view(2, -1, 1, 1, 1))
            self.register_buffer('mask_up', mask.view(2, 1, -1, 1, 1))

            self.W_down = nn.Parameter(torch.empty(2, self.rank, host.in_channels, *host.kernel_size))
            self.W_up = nn.Parameter(torch.empty(2, host.out_channels, self.rank, 1, 1))
            if bias:
                self.bias = nn.Parameter(torch.empty(2, 1, host.out_channels))
            else:
                self.register_parameter('bias', None)

            self.stride = host.stride
            self.padding = host.padding
            self.dilation = host.dilation
            self.groups = host.groups

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
            self.W_down.data = self.W_down.data*self.mask_down
            nn.init.zeros_(self.W_up)
            if self.bias:
                nn.init.zeros_(self.bias)

        def get_weight(self):
            return einsum(self.W_up*self.mask_up, self.W_down*self.mask_down, 'g o r ..., g r i ... -> g o i ...')

        def get_bias(self):
            return self.bias if self.bias else None

        def forward(self, x, weight, bias=None):
            # weight: [G,O,I,K,K]
            weight = rearrange(weight, 'g o i ... -> (g o) i ...')
            x = rearrange(x, '(g b) c w h -> b (g c) w h', g=2)
            x = F.conv2d(x, weight, bias.flatten(), self.stride, self.padding, self.dilation, groups=2)
            x = rearrange(x, 'b (g c) w h -> (g b) c w h', g=2)
            return x

        def get_collapsed_param(self):
            w = einsum(self.W_up.data, self.W_down.data, 'g o r ..., g r i ... -> g o i ...')
            b = self.bias.data if self.bias else None
            return w, b

lora_layer_map = {
    'lora':LoraLayer,
    'dapp':DAPPLayer,
}
