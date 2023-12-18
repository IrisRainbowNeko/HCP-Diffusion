import torch
from torch import nn

class WeightedLoss(nn.Module):
    need_sigma = True

    def __init__(self, loss, weight: float = 1.):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor):
        return self.weight*self.loss(input, target)/sigma**2
