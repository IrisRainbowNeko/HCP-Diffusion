from typing import Union, Tuple

import torch

from .base import SigmaScheduler

class FlowSigmaScheduler(SigmaScheduler):
    def __init__(self, t_start=0, t_end=1):
        super().__init__()
        self.t_start = t_start
        self.t_end = t_end

    def sigma(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor([t])
        t = (self.t_end-self.t_start)*t+self.t_start
        return t

    def alpha(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor([t])
        t = (self.t_end-self.t_start)*t+self.t_start
        return 1-t

    def velocity(self, t: Union[float, torch.Tensor], dt=1e-8, normlize=False) -> Tuple[torch.Tensor, torch.Tensor]:
        r'''
        v(t) = dx(t)/dt = d\alpha(t)/dt * x(0) + d\sigma(t)/dt *eps
        :param t: 0-1, rate of time step
        :return: d\alpha(t)/dt, d\sigma(t)/dt
        '''
        if isinstance(t, float):
            t = torch.tensor([t])
        d_alpha = -torch.ones_like(t)
        d_sigma = torch.ones_like(t)
        if normlize:
            norm = torch.sqrt(d_alpha**2+d_sigma**2)
            return d_alpha/norm, d_sigma/norm
        else:
            return d_alpha, d_sigma

    def alpha_to_t(self, alphas):
        """
        alphas: [B]
        :return: t [B]
        """
        return alphas

    def sigma_to_t(self, sigmas):
        """
        sigmas: [B]
        :return: t [B]
        """
        return 1-sigmas

    def alpha_to_sigma(self, alpha):
        return 1-alpha

    def sigma_to_alpha(self, sigma):
        return 1-sigma

    def c_skip(self, t: Union[float, torch.Tensor]):
        r'''
        \hat{x}(0) = c_skip*x(t) + c_out*f(x(t))
        :param t: 0-1, rate of time step
        '''
        return 1.

    def c_out(self, t: Union[float, torch.Tensor]):
        r'''
        \hat{x}(0) = c_skip*x(t) + c_out*f(x(t))
        :param t: 0-1, rate of time step
        '''
        sigma = self.sigma(t)
        return -sigma
