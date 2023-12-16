import torch
from diffusers import SchedulerMixin
from torch import nn

class MinSNRLoss(nn.MSELoss):
    need_timesteps = True

    def __init__(self, size_average=None, reduce=None, reduction: str = 'none', gamma=1.,
                 noise_scheduler: SchedulerMixin = None, device='cuda:0', **kwargs):
        super().__init__(size_average, reduce, reduction)
        self.gamma = gamma

        # calculate SNR
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0-alphas_cumprod)
        self.alpha = sqrt_alphas_cumprod.to(device)
        self.sigma = sqrt_one_minus_alphas_cumprod.to(device)
        self.all_snr = ((self.alpha/self.sigma)**2).to(device)

    def forward(self, input: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        loss = super(MinSNRLoss, self).forward(input, target)
        snr = self.all_snr[timesteps[:loss.shape[0], ...].squeeze()]
        snr_weight = (self.gamma/snr).clip(max=1.).float()
        return loss*snr_weight.view(-1, 1, 1, 1)


class SoftMinSNRLoss(MinSNRLoss):
    # gamma=2

    def forward(self, input: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        loss = super(MinSNRLoss, self).forward(input, target)
        snr = self.all_snr[timesteps[:loss.shape[0], ...].squeeze()]
        snr_weight = (self.gamma**3/(snr**2 + self.gamma**3)).float()
        return loss*snr_weight.view(-1, 1, 1, 1)

class KDiffMinSNRLoss(MinSNRLoss):

    def forward(self, input: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        loss = super(MinSNRLoss, self).forward(input, target)
        snr = self.all_snr[timesteps[:loss.shape[0], ...].squeeze()]
        snr_weight = 4*(((self.gamma*snr)**2/(snr**2 + self.gamma**2)**2)).float()
        return loss*snr_weight.view(-1, 1, 1, 1)

class EDMLoss(MinSNRLoss):

    def forward(self, input: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        loss = super(MinSNRLoss, self).forward(input, target)
        sigma = self.sigma[timesteps[:loss.shape[0], ...].squeeze()]
        snr = self.all_snr[timesteps[:loss.shape[0], ...].squeeze()]
        snr_weight = ((sigma**2+self.gamma**2)/(snr*(sigma*self.gamma)**2)).float()
        return loss*snr_weight.view(-1, 1, 1, 1)