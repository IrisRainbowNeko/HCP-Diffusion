import torch

from .base import BaseSampler
from .sigma_scheduler import SigmaScheduler

class EDM_DDPMSampler(BaseSampler):
    def __init__(self, sigma_scheduler: SigmaScheduler, generator: torch.Generator = None, sigma_thr=1000):
        super().__init__(sigma_scheduler, generator)
        self.sigma_thr = sigma_thr

    def c_in(self, sigma):
        return 1/(sigma**2+1).sqrt()

    def c_out(self, sigma):
        return -sigma

    def c_skip(self, sigma):
        return 1.

    def add_noise(self, x, sigma):
        x = x.clone()
        x[sigma.view(-1)>self.sigma_thr, ...] = 0.
        return x+sigma.view(-1, 1, 1, 1)*self.make_nosie(x.shape, device=x.device, dtype=x.dtype)

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError

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

    def add_noise(self, x, sigma):
        x = x.clone()
        x[sigma.view(-1)>self.sigma_thr, ...] = 0.
        return x+sigma.view(-1, 1, 1, 1)*self.make_nosie(x.shape, device=x.device, dtype=x.dtype)

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError
