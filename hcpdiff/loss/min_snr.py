import torch

from .weighting import WeightedLoss

class MinSNRWeight(WeightedLoss):
    def __init__(self, loss, weight: float = 1., gamma: float = 1.):
        super().__init__(loss, weight)
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        loss = self.loss(input, target)
        snr = 1/(sigma*torch.sqrt(1+sigma**2))
        snr_weight = snr.clip(max=self.gamma).float()
        return self.weight*loss*snr_weight.view(-1, 1, 1, 1)

class SoftMinSNRWeight(MinSNRWeight):

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        loss = self.loss(input, target)
        snr_weight = (self.gamma**2/(sigma**2+self.gamma**2)).float()
        return self.weight*loss*snr_weight.view(-1, 1, 1, 1)

class KDiffMinSNRWeight(MinSNRWeight):

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        loss = self.loss(input, target)
        snr_weight = ((self.gamma*sigma)**2/(sigma**2+self.gamma**2)**2).float()
        return self.weight*loss*snr_weight.view(-1, 1, 1, 1)

class EDMWeight(MinSNRWeight):

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        loss = self.loss(input, target)
        snr_weight = ((sigma**2+self.gamma**2)/((sigma*self.gamma)**2)).float()
        return self.weight*loss*snr_weight.view(-1, 1, 1, 1)
