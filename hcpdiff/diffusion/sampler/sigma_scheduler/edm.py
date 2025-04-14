from typing import Union

import torch
import numpy as np

from .base import SigmaScheduler

class EDMSigmaScheduler(SigmaScheduler):
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0, num_timesteps=1000):
        self.sigma_min = torch.tensor(sigma_min)
        self.sigma_max = torch.tensor(sigma_max)
        self.rho = rho

        self.num_timesteps=num_timesteps

    def get_sigma(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor(t)

        min_inv_rho = self.sigma_min**(1/self.rho)
        max_inv_rho = self.sigma_max**(1/self.rho)
        return torch.lerp(min_inv_rho, max_inv_rho, t)**self.rho

    def sample_sigma(self, min_rate=0.0, max_rate=1.0, shape=(1,)):
        if isinstance(min_rate, float):
            min_rate = torch.full(shape, min_rate)
        if isinstance(max_rate, float):
            max_rate = torch.full(shape, max_rate)

        t = torch.lerp(min_rate, max_rate, torch.rand_like(min_rate))
        return self.get_sigma(t), t

class EDMRefSigmaScheduler(EDMSigmaScheduler):
    def __init__(self, ref_scheduler, sigma_min=0.002, sigma_max=80.0, rho=7.0, num_timesteps=1000):
        super().__init__(sigma_min, sigma_max, rho, num_timesteps=num_timesteps)
        self.ref_sigmas = ref_scheduler.sigmas.cpu().clip(min=1e-8).log().numpy()
        self.ref_t = np.linspace(0, 1, len(self.ref_sigmas))

    def sample_sigma(self, min_rate=0.0, max_rate=1.0, shape=(1,)):
        if isinstance(min_rate, float):
            min_rate = torch.full(shape, min_rate)
        if isinstance(max_rate, float):
            max_rate = torch.full(shape, max_rate)

        t = torch.lerp(min_rate, max_rate, torch.rand_like(min_rate))
        sigma = self.get_sigma(t)
        t_rect = torch.tensor(np.interp(sigma.cpu().clip(min=1e-8).log().numpy(), self.ref_sigmas, self.ref_t))
        return sigma, t_rect