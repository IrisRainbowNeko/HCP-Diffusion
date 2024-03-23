import torch

from .base import BaseSampler
from .sigma_scheduler import SigmaScheduler

class DDPMSampler(BaseSampler):
    def __init__(self, sigma_scheduler: SigmaScheduler, generator: torch.Generator=None):
        super().__init__(sigma_scheduler, generator)

    def c_in(self, sigma):
        return 1.

    def c_out(self, sigma):
        return -sigma

    def c_skip(self, sigma):
        return (sigma**2+1).sqrt()  # 1/sqrt(alpha_)

    def add_noise(self, x, sigma):
        sqrt_alpha = 1./(sigma**2+1).sqrt()
        one_sqrt_alpha = (1-sqrt_alpha**2).sqrt()
        return sqrt_alpha*x+one_sqrt_alpha*self.make_nosie(x.shape, device=x.device, dtype=x.dtype)

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError
