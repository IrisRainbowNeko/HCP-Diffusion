import torch

from .base import BaseSampler
from .sigma_scheduler import SigmaScheduler

class EDMSampler(BaseSampler):
    def __init__(self, sigma_scheduler: SigmaScheduler, generator: torch.Generator = None, sigma_data: float = 1.0, sigma_thr=1000):
        super().__init__(sigma_scheduler, generator)
        self.sigma_data = sigma_data
        self.sigma_thr = sigma_thr

    def c_in(self, sigma):
        return 1/(sigma**2+self.sigma_data**2).sqrt()

    def c_out(self, sigma):
        return (sigma*self.sigma_data)/(sigma**2+self.sigma_data**2).sqrt()

    def c_skip(self, sigma):
        return self.sigma_data**2/(sigma**2+self.sigma_data**2)

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError
