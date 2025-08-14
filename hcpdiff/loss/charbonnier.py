import torch
from torch import nn

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3, size_average=True):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.size_average = size_average

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt((diff * diff) + (self.eps*self.eps))
        if self.size_average:
            loss = loss.mean()
        return loss