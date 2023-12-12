import torch
from diffusers import SchedulerMixin, DDPMScheduler, UNet2DConditionModel

from .base import BaseSampler

class DiffusersSampler(BaseSampler):
    def __init__(self, generator: torch.Generator, scheduler: SchedulerMixin):
        super().__init__(generator)
        self.scheduler = scheduler

    def c_in(self, sigma):
        one = torch.FloatTensor(1.)
        if hasattr(self.scheduler, '_step_index'):
            self.scheduler._step_index = None
        return self.scheduler.scale_model_input(one, sigma)

    def c_out(self, sigma):
        return -sigma

    def c_skip(self, sigma):
        if self.c_in(sigma) == 1.:  # DDPM model
            return (sigma**2+1).sqrt()  # 1/sqrt(alpha_)
        else:  # EDM model
            return 1.

    def init_noise(self, sigma, shape, device='cuda', dtype=torch.float32):
        return torch.randn(shape, generator=self.generator, device=device, dtype=dtype)*self.scheduler.init_noise_sigma

    def add_noise(self, x, sigma, t=None):
        noise = torch.randn(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)
        return self.scheduler.add_noise(x, noise, t)

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError
