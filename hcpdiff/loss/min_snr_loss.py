import torch
from diffusers import SchedulerMixin
from torch import nn

class MinSNRLoss(nn.MSELoss):
    need_sigma = True

    def __init__(self, size_average=None, reduce=None, reduction: str = 'none', gamma=1., **kwargs):
        super().__init__(size_average, reduce, reduction)
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        loss = super(MinSNRLoss, self).forward(input, target)
        snr = 1/sigma
        snr_weight = snr.clip(max=self.gamma).float()
        return loss*snr_weight.view(-1, 1, 1, 1)


class SoftMinSNRLoss(MinSNRLoss):

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        loss = super(MinSNRLoss, self).forward(input, target)
        snr = 1/sigma
        snr_weight = (self.gamma**3/(snr**2 + self.gamma**3)).float()
        return loss*snr_weight.view(-1, 1, 1, 1)

class KDiffMinSNRLoss(MinSNRLoss):

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        loss = super(MinSNRLoss, self).forward(input, target)
        snr = 1/sigma
        snr_weight = 4*(((self.gamma*snr)**2/(snr**2 + self.gamma**2)**2)).float()
        return loss*snr_weight.view(-1, 1, 1, 1)

class EDMLoss(MinSNRLoss):

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        loss = super(MinSNRLoss, self).forward(input, target)
        snr = 1/sigma
        snr_weight = ((sigma**2+self.gamma**2)/(snr*(sigma*self.gamma)**2)).float()
        return loss*snr_weight.view(-1, 1, 1, 1)