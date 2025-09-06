from typing import Union, Tuple

import numpy as np
import torch

from .base import SigmaScheduler

class EDMSigmaScheduler(SigmaScheduler):
    def __init__(self, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

    def sigma_edm(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        if isinstance(t, float):
            t = torch.tensor(t)

        min_inv_rho = self.sigma_min**(1/self.rho)
        max_inv_rho = self.sigma_max**(1/self.rho)
        return torch.lerp(min_inv_rho, max_inv_rho, t)**self.rho

    def sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        r'''
        x_t = c_in(t) * (x(0) + \sigma(t)*eps), eps~N(0,I)
        '''
        if isinstance(t, float):
            t = torch.tensor(t)

        sigma_edm = self.sigma_edm(t)
        return sigma_edm/torch.sqrt(sigma_edm**2+self.sigma_data**2)

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        r'''
        x_t = c_in(t) * (x(0) + \sigma(t)*eps), eps~N(0,I)
        '''
        if isinstance(t, float):
            t = torch.tensor(t)

        sigma_edm = self.sigma_edm(t)
        return 1./torch.sqrt(sigma_edm**2+self.sigma_data**2)

    def c_skip(self, t: Union[float, torch.Tensor]):
        r'''
        \hat{x}(0) = c_skip(t)*(x(t)/c_in(t)) + c_out(t)*f(x(t))
        :param t: 0-1, rate of time step
        '''
        sigma_edm = self.sigma_edm(t)
        return self.sigma_data**2/torch.sqrt(sigma_edm**2+self.sigma_data**2)

    def c_out(self, t: Union[float, torch.Tensor]):
        r'''
        \hat{x}(0) = c_skip(t)*(x(t)/c_in(t)) + c_out(t)*f(x(t))
        :param t: 0-1, rate of time step
        '''
        sigma_edm = self.sigma_edm(t)
        return (self.sigma_data*sigma_edm)/torch.sqrt(sigma_edm**2+self.sigma_data**2)

    def c_noise(self, t: Union[float, torch.Tensor]):
        sigma_edm = self.sigma_edm(t)
        return sigma_edm.log()/4

    @property
    def sigma_start(self):
        return self.sigma(0)

    @property
    def sigma_end(self):
        return self.sigma(1)

    @property
    def alpha_start(self):
        return self.alpha(0)

    @property
    def alpha_end(self):
        return self.alpha(1)

    def alpha_to_sigma(self, alpha):
        return torch.sqrt(1 - (alpha*self.sigma_data)**2)

    def sigma_to_alpha(self, sigma):
        return torch.sqrt(1 - sigma**2)/self.sigma_data

class EDMTimeRescaleScheduler(EDMSigmaScheduler):
    def __init__(self, ref_scheduler: SigmaScheduler, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        super().__init__(sigma_min, sigma_max, rho)
        self.ref_scheduler = ref_scheduler

    def scale_t(self, t):
        ref_t = torch.linspace(0, 1, 1000)
        alphas = self.alpha(ref_t)
        sigmas = self.sigma(ref_t)
        sigmas_edm = sigmas/alphas
        sigma_edm = self.sigma_edm(t)
        t = np.interp(sigma_edm.cpu().clip(min=1e-8).log().numpy(), sigmas_edm, ref_t.numpy())
        return torch.tensor(t)

    def sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        return self.ref_scheduler.sigma(t)

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        return self.ref_scheduler.alpha(t)

    def velocity(self, t: Union[float, torch.Tensor], dt=1e-8, normlize=True) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ref_scheduler.velocity(t, dt=dt, normlize=normlize)

    def c_skip(self, t: Union[float, torch.Tensor]):
        return self.ref_scheduler.c_skip(t)

    def c_out(self, t: Union[float, torch.Tensor]):
        return self.ref_scheduler.c_out(t)
    
    def c_noise(self, t: Union[float, torch.Tensor]):
        return self.ref_scheduler.c_noise(t)

    def sample(self, min_t=0.0, max_t=1.0, shape=(1,)):
        if isinstance(min_t, float):
            min_t = torch.full(shape, min_t)
        if isinstance(max_t, float):
            max_t = torch.full(shape, max_t)

        t = torch.lerp(min_t, max_t, torch.rand_like(min_t))
        t = self.scale_t(t)
        return t
