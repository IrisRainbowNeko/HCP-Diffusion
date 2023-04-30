import torch
from torch import nn
from diffusers import SchedulerMixin

class MinSNRLoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'none', gamma=1.,
                 noise_scheduler: SchedulerMixin=None, device='cuda:0', **kwargs):
        super().__init__(size_average, reduce, reduction)
        self.gamma=gamma

        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        alpha = sqrt_alphas_cumprod
        sigma = sqrt_one_minus_alphas_cumprod
        self.all_snr = ((alpha / sigma) ** 2).to(device)

    def forward(self, input: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        loss = super(MinSNRLoss, self).forward(input, target)
        snr_weight = (self.gamma/self.all_snr[timesteps[:loss.shape[0], ...].squeeze()]).clip(max=1.).float()
        return loss * snr_weight.view(-1,1,1,1)