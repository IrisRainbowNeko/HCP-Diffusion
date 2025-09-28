from typing import Union

import torch

from .base import SigmaScheduler

class EDMSigmaScheduler(SigmaScheduler):
    def __init__(self, timer, t_shifter=tuple(), sigma_data=0.5):
        super().__init__(timer, t_shifter)
        self.sigma_data = sigma_data

    def sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        r'''
        x_t = c_in(t) * (x(0) + \sigma(t)*eps), eps~N(0,I)
        '''
        if isinstance(t, float):
            t = torch.tensor(t)

        sigma_edm = self.shift(t)
        return sigma_edm/torch.sqrt(sigma_edm**2+self.sigma_data**2)

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        r'''
        x_t = c_in(t) * (x(0) + \sigma(t)*eps), eps~N(0,I)
        '''
        if isinstance(t, float):
            t = torch.tensor(t)

        sigma_edm = self.shift(t)
        return 1./torch.sqrt(sigma_edm**2+self.sigma_data**2)

    def c_skip(self, t: Union[float, torch.Tensor]):
        r'''
        \hat{x}(0) = c_skip(t)*(x(t)/c_in(t)) + c_out(t)*f(x(t))
        :param t: 0-1, rate of time step
        '''
        sigma_edm = self.shift(t)
        return self.sigma_data**2/torch.sqrt(sigma_edm**2+self.sigma_data**2)

    def c_out(self, t: Union[float, torch.Tensor]):
        r'''
        \hat{x}(0) = c_skip(t)*(x(t)/c_in(t)) + c_out(t)*f(x(t))
        :param t: 0-1, rate of time step
        '''
        sigma_edm = self.shift(t)
        return (self.sigma_data*sigma_edm)/torch.sqrt(sigma_edm**2+self.sigma_data**2)

    def c_noise(self, t: Union[float, torch.Tensor]):
        sigma_edm = self.shift(t)
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
