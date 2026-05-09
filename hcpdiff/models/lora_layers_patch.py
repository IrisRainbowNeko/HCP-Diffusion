"""
lora_layers.py
====================
    :Name:        lora layers
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     09/04/2023
    :Licence:     Apache-2.0
"""

import math

import torch
from einops import einsum
from torch import nn
from torch.nn import functional as F

from .lora_base_patch import LoraBlock, PatchPluginContainer, OFTBlock

class LoraLayer(LoraBlock):
    def __init__(self, name: str, host, rank=1, dropout=0.0, alpha=1.0, bias=False, alpha_auto_scale=True, **kwargs):
        super().__init__(name, host, rank, dropout, alpha=alpha, bias=bias, alpha_auto_scale=alpha_auto_scale, **kwargs)

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

def none_add(a, b):
    if a is None:
        return b
    return a+b

def project(R, eps=1e-5):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0], dtype=R.dtype, device=R.device))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(diff, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    return torch.where(mask, R, I + eps * (diff / norm_diff))

def cayley(data, is_linear=True):
    r, c = list(data.shape)
    # Ensure the input matrix is skew-symmetric
    skew = 0.5 * (data - data.t())
    I = torch.eye(r, device=data.device)
    
    # Perform the Cayley parametrization
    if is_linear:
        Q = torch.mm(I + skew, torch.inverse(I - skew))
    else:
        Q = torch.mm(I - skew, torch.inverse(I + skew))
    return Q

def cayley_batch(data):
    b, r, c = data.shape
    # Ensure the input matrix is skew-symmetric
    skew = 0.5 * (data - data.transpose(1, 2))
    I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

    # Perform the Cayley parametrization
    Q = torch.bmm(I - skew, torch.inverse(I + skew))

    return Q


class OFTLayer(OFTBlock): 
    def __init__(self, name: str, host, r=4, eps=1e-5, is_coft=True, block_share=False, 
                 dropout=0.0, alpha=1.0, bias=False, alpha_auto_scale=True, **kwargs):
        super().__init__(name, host, r=r, eps=eps, dropout=dropout, is_coft=is_coft, block_share=block_share,
                         alpha=alpha, bias=bias, alpha_auto_scale=alpha_auto_scale, **kwargs)

    class LinearLayer(OFTBlock.LinearLayer):
        def __init__(self, host: nn.Linear, bias, block, r=4, eps=1e-5, is_coft=True, block_share=False):
            super().__init__(host, r, bias, block)
            assert host.in_features % r == 0, "in_features must be divisible by r"

            self.in_features = host.in_features
            self.out_features = host.out_features
            self.block_share = block_share

            # self.OFT = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=bias)
            # self.register_buffer("OFT_weight", host.weight.detach().clone())
            # self.register_buffer("OFT_bias", host.bias)

            if block_share:
                self.R_shape = [self.in_features // r, self.in_features // r]
                self.R = nn.Parameter(torch.zeros(self.R_shape[0], self.R_shape[0]))
                self.eps = eps * self.R_shape[0] ** 2
            else:
                self.R_shape = [r, self.in_features // r, self.in_features // r]
                R = torch.zeros(self.R_shape[1], self.R_shape[1])
                R = torch.stack([R.clone() for _ in range(r)])
                self.R = nn.Parameter(R)
                self.eps = eps * self.R_shape[1] ** 2

        def reset_parameters(self):
            with torch.no_grad():
                self.R.zero_()

        def forward(self, x, weight=None, bias=None):
            # out = nn.functional.linear(input=x, weight=weight, bias=bias)
            # return out
            x_shape = x.shape
            if bias is None:
                return torch.mm(x.view(-1, x_shape[-1]), weight.transpose(0, 1)).view(*x_shape[:-1], -1)
            else:
                return torch.mm(x.view(-1, x_shape[-1]), weight.transpose(0, 1)).view(*x_shape[:-1], -1) + bias

        
        def get_bias(self):
            bias = self.OFT_bias.data if self.OFT_bias is not None else None
            return bias
        
        def get_weight(self):
            if self.block_share:
                R_rot = cayley(self.R)
                blocks = [R_rot] * self.r
            else:
                R_rot = cayley_batch(self.R)
                blocks = [R_rot[i] for i in range(self.r)]

            orth_matrix = torch.block_diag(*blocks)
            fix_filt = self.OFT_weight.data
            fix_filt = torch.transpose(fix_filt, 0, 1).to(self.R.dtype)
            filt = torch.mm(orth_matrix, fix_filt)
            filt = torch.transpose(filt, 0, 1)
            return filt

        def get_collapsed_param(self):
            if self.block_share:
                R_rot = cayley(self.R)
                blocks = [R_rot] * self.r
            else:
                R_rot = cayley_batch(self.R)
                blocks = [R_rot[i] for i in range(self.r)]

            orth_matrix = torch.block_diag(*blocks)
            fix_filt = self.OFT_weight.data
            # fix_filt = self.OFT.data
            fix_filt = torch.transpose(fix_filt, 0, 1).to(self.R.dtype)
            filt = torch.mm(orth_matrix, fix_filt)
            filt = torch.transpose(filt, 0, 1)
            bias = self.OFT_bias.data if self.OFT_bias is not None else None

            return filt, bias
        
        def is_orthogonal(self, R, eps=1e-5):
            with torch.no_grad():
                RtR = torch.matmul(R.t(), R)
                diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
                return torch.all(diff < eps)

        def is_identity_matrix(self, tensor):
            if not torch.is_tensor(tensor):
                raise TypeError("Input must be a PyTorch tensor.")
            if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
                return False
            identity = torch.eye(tensor.shape[0], device=tensor.device)
            return torch.all(torch.eq(tensor, identity))

    class Conv2dLayer(OFTBlock.Conv2dLayer):
        def __init__(self, host: nn.Conv2d, bias, block, r=4, eps=1e-5, is_coft=True, block_share=False):
            super().__init__(host, r, bias, block)
            assert host.in_channels % r == 0, "in_channels must be divisible by r"

            self.in_channels = host.in_channels
            self.out_channels = host.out_channels
            self.kernel_size = host.kernel_size
            self.stride = host.stride
            self.padding = host.padding
            self.dilation = host.dilation
            self.groups = host.groups
            self.r = r
            self.is_coft = is_coft
            self.block_share = block_share
            self.eps = eps

            # fix conv-kernel
            # self.OFT = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, 
            #                      padding=self.padding, bias=bias)

            # self.register_buffer("OFT_weight", host.weight.detach().clone())
            # self.register_buffer("OFT_bias", host.bias)

            self.filt_shape = [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size]
            self.fix_filt_shape = [self.kernel_size * self.kernel_size * self.in_channels, self.out_channels]

            if block_share:
                self.R_shape = [self.in_channels // r, self.in_channels // r]
                self.R = nn.Parameter(torch.zeros(self.R_shape[0], self.R_shape[0]))
                self.eps = eps * self.R_shape[0] ** 2
            else:
                self.R_shape = [r, self.in_channels // r, self.in_channels // r]
                R = torch.zeros(self.R_shape[1], self.R_shape[1])
                R = torch.stack([R.clone() for _ in range(r)])
                self.R = nn.Parameter(R)
                self.eps = eps * self.R_shape[1] ** 2

        def reset_parameters(self):
            with torch.no_grad():
                self.R.zero_()

        def forward(self, x, weight=None, bias=None):
            out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride, padding=self.padding)
            return out

        def get_collapsed_param(self):
            fix_filt = self.OFT_weight.data
            fix_filt = fix_filt.view(self.fix_filt_shape)
            if self.block_share:
                R_rot = cayley(self.R)
                blocks = [R_rot] * self.r
            else:
                R_rot = cayley_batch(self.R)
                blocks = [R_rot[i] for i in range(self.r)]

            orth_matrix = torch.block_diag(*blocks)
            filt = torch.mm(orth_matrix, fix_filt)
            filt = filt.view(self.filt_shape)
            bias = self.OFT_bias.data if self.OFT_bias is not None else None
            return filt, bias
        
        def get_bias(self):
            bias = self.OFT_bias.data if self.OFT_bias is not None else None
            return bias
        
        def get_weight(self):
            fix_filt = self.OFT_weight.data
            fix_filt = fix_filt.view(self.fix_filt_shape)
            if self.block_share:
                R_rot = cayley(self.R)
                blocks = [R_rot] * self.r
            else:
                R_rot = cayley_batch(self.R)
                blocks = [R_rot[i] for i in range(self.r)]

            orth_matrix = torch.block_diag(*blocks)
            filt = torch.mm(orth_matrix, fix_filt)
            filt = filt.view(self.filt_shape)
            return filt
        
        def is_orthogonal(self, R, eps=1e-5):
            with torch.no_grad():
                RtR = torch.matmul(R.t(), R)
                diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
                return torch.all(diff < eps)
                
class DAPPPatchContainer(PatchPluginContainer):
    def forward(self, x, *args, **kwargs):
        weight_p = None
        weight_n = None
        bias_p = None
        bias_n = None
        for name in self.plugin_names:
            if self[name].branch=='p':
                weight_p = none_add(weight_p, self[name].get_weight())
                bias_p = none_add(bias_p, self[name].get_bias())
            elif self[name].branch=='n':
                weight_n = none_add(weight_n, self[name].get_weight())
                bias_n = none_add(bias_n, self[name].get_bias())

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
    'oft':OFTLayer,
    'dapp':DAPPLayer,
}
