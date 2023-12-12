import torch
from diffusers import SchedulerMixin
from .noise_base import NoiseBase

class ZeroTerminalScheduler(NoiseBase, SchedulerMixin):
    def __init__(self, base_scheduler):
        super().__init__(base_scheduler)
        base_scheduler.betas = self.rescale_zero_terminal_snr(base_scheduler.betas)
        base_scheduler.alphas = 1.0-base_scheduler.betas
        base_scheduler.alphas_cumprod = torch.cumprod(base_scheduler.alphas, dim=0)

    def rescale_zero_terminal_snr(self, betas):
        """
        Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
        Args:
            betas (`torch.FloatTensor`):
                the betas that the scheduler is being initialized with.
        Returns:
            `torch.FloatTensor`: rescaled betas with zero terminal SNR
        """
        # Convert betas to alphas_bar_sqrt
        alphas = 1.0-betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = alphas_cumprod.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0/(alphas_bar_sqrt_0-alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
        alphas = alphas_bar[1:]/alphas_bar[:-1]  # Revert cumprod
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1-alphas

        return betas


