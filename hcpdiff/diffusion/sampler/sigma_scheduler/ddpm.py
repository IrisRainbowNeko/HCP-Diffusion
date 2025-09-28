from typing import Union, Tuple

import torch

from .base import SigmaScheduler
from ..shifter import DDPMDiscreteShifter, DDPMContinuousShifter
from ..timer import Timer

class DDPMSigmaScheduler(SigmaScheduler):
    def __init__(self, timer=None, t_shifter=None, pred_type='eps'):
        super().__init__(timer, t_shifter=DDPMDiscreteShifter() if t_shifter is None else tuple())
        self.pred_type = pred_type

    def sigma(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor(t)
        alpha_cumprod = self.shift(t)
        return (1-alpha_cumprod).sqrt()

    def alpha(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor(t)
        alpha_cumprod = self.shift(t)
        return alpha_cumprod.sqrt()

    def c_noise(self, t: Union[float, torch.Tensor]):
        return t*1000

    def velocity(self, t: Union[float, torch.Tensor], dt=1e-3, normlize=True) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        v(t) = dx(t)/dt = d\alpha(t)/dt * x(0) + d\sigma(t)/dt *eps
        :param t: 0-1, rate of time step
        :return: d\alpha(t)/dt, d\sigma(t)/dt
        '''
        if normlize:
            d_alpha = -self.sigma(t)
            d_sigma = self.alpha(t)
            return d_alpha, d_sigma
        else:
            if isinstance(t, float):
                t = torch.tensor(t)
            ac_t = self.shift(t)
            ac_t1 = self.shift(t-dt)

            d_alpha = (ac_t.sqrt()-ac_t1.sqrt())/(ac_t-ac_t1)
            d_sigma = ((1-ac_t).sqrt()-(1-ac_t1).sqrt())/(ac_t-ac_t1)
            return d_alpha, d_sigma

    def alpha_to_sigma(self, alpha):
        return torch.sqrt(1-alpha**2)

    def sigma_to_alpha(self, sigma):
        return torch.sqrt(1-sigma**2)

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    sigma_scheduler = DDPMSigmaScheduler()
    sigma_scheduler_c = DDPMSigmaScheduler()
    sigma_scheduler_c.set_timer(Timer(shifter=DDPMContinuousShifter()))

    t = torch.linspace(0, 1, 1000)
    alphas = sigma_scheduler.alpha(t)
    sigmas = sigma_scheduler.sigma(t)
    alphas_c = sigma_scheduler_c.alpha(t)

    plt.figure()
    plt.plot(t*1000, (alphas_c**2).log())
    plt.plot(t*1000, (alphas**2).log())
    plt.show()
