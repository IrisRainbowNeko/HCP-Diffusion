import torch
from torch import nn


class MSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'none', **kwargs):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return super(MSELoss, self).forward(input, target)