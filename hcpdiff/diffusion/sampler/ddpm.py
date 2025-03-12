import torch

from .base import BaseSampler
from .sigma_scheduler import SigmaScheduler

class DDPMSampler(BaseSampler):
    def __init__(self, sigma_scheduler: SigmaScheduler, generator: torch.Generator=None):
        super().__init__(sigma_scheduler, generator)

    def c_in(self, sigma):
        return 1./(sigma**2+1).sqrt()

    def c_out(self, sigma):
        return -sigma

    def c_skip(self, sigma):
        return 1.

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError
