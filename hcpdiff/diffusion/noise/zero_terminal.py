import torch
from ..sampler.sigma_scheduler import DDPMDiscreteSigmaScheduler

class ZeroTerminalSampler:

    @classmethod
    def patch(cls, base_sampler):
        assert isinstance(base_sampler.sigma_scheduler, DDPMDiscreteSigmaScheduler), "ZeroTerminalScheduler only works with DDPM SigmaScheduler"

        alphas_cumprod = base_sampler.sigma_scheduler.alphas_cumprod
        base_sampler.sigma_scheduler.alphas_cumprod = cls.rescale_zero_terminal_snr(alphas_cumprod)
        base_sampler.sigma_scheduler.sigmas = ((1-alphas_cumprod)/alphas_cumprod).sqrt()


    @staticmethod
    def rescale_zero_terminal_snr(alphas_cumprod, thr=1e-4):
        """
        Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
        Args:
            alphas_cumprod (`torch.FloatTensor`)
        Returns:
            `torch.FloatTensor`: rescaled betas with zero terminal SNR
        """
        alphas_bar_sqrt = alphas_cumprod.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0/(alphas_bar_sqrt_0-alphas_bar_sqrt_T)
        alphas_bar_sqrt[-1] = thr # avoid nan sigma

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
        return alphas_bar
