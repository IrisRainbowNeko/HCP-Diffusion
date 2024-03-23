import torch

from .base import BaseSampler
from .sigma_scheduler import SigmaScheduler

class EDM_DDPMSampler(BaseSampler):
    def __init__(self, sigma_scheduler: SigmaScheduler, generator: torch.Generator = None):
        super().__init__(sigma_scheduler, generator)

    def c_in(self, sigma):
        return 1/(sigma**2+1).sqrt()

    def c_out(self, sigma):
        return -sigma

    def c_skip(self, sigma):
        return 1.

    def add_noise(self, x, sigma):
        return x+sigma*torch.randn(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError

class EDMSampler(BaseSampler):
    def __init__(self, sigma_scheduler: SigmaScheduler, generator: torch.Generator = None, sigma_data: float = 0.5):
        super().__init__(sigma_scheduler, generator)
        self.sigma_data = sigma_data

    def c_in(self, sigma):
        return 1/(sigma**2+self.sigma_data**2)**0.5

    def c_out(self, sigma):
        return sigma*self.sigma_data/(sigma**2+self.sigma_data**2)**0.5

    def c_skip(self, sigma):
        return self.sigma_data**2/(sigma**2+self.sigma_data**2)

    def add_noise(self, x, sigma):
        return x+sigma.view(-1,1,1,1)*torch.randn(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError