"""
layers.py
====================
    :Name:        GroupLinear and other layers
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     09/04/2023
    :Licence:     Apache-2.0
"""

import torch
from torch import nn
import math
from einops import rearrange

class GroupLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, groups: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        assert in_features%groups == 0
        assert out_features%groups == 0

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.groups = groups
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((groups, in_features//groups, out_features//groups), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(groups, 1, out_features//groups, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        self.kaiming_uniform_group(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _calculate_fan_in_and_fan_out(tensor):
        receptive_field_size = 1
        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    @staticmethod
    def kaiming_uniform_group(tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> torch.Tensor:
        def _calculate_correct_fan(tensor, mode):
            mode = mode.lower()
            valid_modes = ['fan_in', 'fan_out']
            if mode not in valid_modes:
                raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

            fan_in, fan_out = GroupLinear._calculate_fan_in_and_fan_out(tensor)
            return fan_in if mode == 'fan_in' else fan_out

        fan = _calculate_correct_fan(tensor, mode)
        gain = nn.init.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor): # x: [B,G,L,C]
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out