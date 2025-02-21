import torch

from .weighting import WeightedLoss

class MinSNRWeight(WeightedLoss):
    def __init__(self, loss, loss_type='eps', weight: float = 1., gamma: float = 1.):
        super().__init__(loss, weight)
        self.gamma = gamma
        self.loss_type = loss_type

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        loss = self.loss(input, target)
        if self.loss_type=='x0':
            w_snr = (1/(sigma**2)).clip(max=self.gamma).float()
        elif self.loss_type=='eps':
            w_snr = (self.gamma*sigma**2).clip(max=1).float()
        return self.weight*loss*w_snr.view(-1, 1, 1, 1)

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
