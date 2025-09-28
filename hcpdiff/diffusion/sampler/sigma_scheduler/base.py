from typing import Union, Tuple, Callable, List

import torch

from ..timer import Timer

shifter_type = Callable[[torch.Tensor, ...], torch.Tensor]

class SigmaScheduler:
    def __init__(self, timer: Timer = None, t_shifter: List[shifter_type] | shifter_type = tuple()):
        if not isinstance(t_shifter, (list, tuple)):
            t_shifter = [t_shifter]

        self.timer = timer or Timer()
        self.t_shifter = t_shifter
        self.states = {}

    def set_states(self, **states):
        self.states = states

    def update_states(self, **states):
        self.states.update(states)

    def shift(self, t, shape=(1,), **kwargs) -> torch.Tensor:
        '''
        :return: t: Real timesteps.
                 st: Shifted timesteps for diffusion process. e.g. DDPM: x_t = \sqrt(st)*x_0 + \sqrt(1-st)*eps
        '''
        if isinstance(t, float):
            t = torch.full(shape, t)

        st = t
        for shifter in self.t_shifter:
            st = shifter(st, **self.states, **kwargs)
        return st

    @property
    def min_dt(self):
        min_dts = [(shifter.min_dt if hasattr(shifter, 'min_dt') else 1e-8) for shifter in self.t_shifter]+[1e-8]
        return max(min_dts)

    def sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        r'''
        x(t) = \alpha(t)*x(0) + \sigma(t)*eps
        :param t: 0-1, rate of time step
        '''
        raise NotImplementedError

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        r'''
        x(t) = \alpha(t)*x(0) + \sigma(t)*eps
        :param t: 0-1, rate of time step
        '''
        raise NotImplementedError

    def velocity(self, t: Union[float, torch.Tensor], dt=1e-8, normlize=True) -> Tuple[torch.Tensor, torch.Tensor]:
        r'''
        v(t) = dx(t)/dt = d\alpha(t)/dt * x(0) + d\sigma(t)/dt *eps
        :param t: 0-1, rate of time step
        :return: d\alpha(t)/dt, d\sigma(t)/dt
        '''
        d_alpha = (self.alpha(t)-self.alpha(t-dt))/dt
        d_sigma = (self.sigma(t)-self.sigma(t-dt))/dt
        if normlize:
            norm = torch.sqrt(d_alpha**2+d_sigma**2)
            return d_alpha/norm, d_sigma/norm
        else:
            return d_alpha, d_sigma

    @property
    def sigma_start(self):
        return self.sigma(torch.tensor(0.))

    @property
    def sigma_end(self):
        return self.sigma(torch.tensor(1.))

    @property
    def alpha_start(self):
        return self.alpha(torch.tensor(0.))

    @property
    def alpha_end(self):
        return self.alpha(torch.tensor(1.))

    def alpha_to_sigma(self, alpha):
        raise NotImplementedError

    def sigma_to_alpha(self, sigma):
        raise NotImplementedError

    def c_in(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            return 1.
        else:
            return torch.ones_like(t, dtype=torch.float32)

    def c_skip(self, t: Union[float, torch.Tensor]):
        r'''
        \hat{x}(0) = c_skip*x(t) + c_out*f(x(t))
        :param t: 0-1, rate of time step
        '''
        return 1./self.alpha(t)

    def c_out(self, t: Union[float, torch.Tensor]):
        r'''
        \hat{x}(0) = c_skip*x(t) + c_out*f(x(t))
        :param t: 0-1, rate of time step
        '''
        return -self.sigma(t)/self.alpha(t)

    def c_noise(self, t: Union[float, torch.Tensor]):
        return self.shift(t)*1000.

class TimeSigmaScheduler(SigmaScheduler):
    def __init__(self, timer=None, num_timesteps=1000):
        super().__init__(timer)
        self.num_timesteps = num_timesteps

    def sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        '''
        :param t: 0-1, rate of time step
        '''
        if isinstance(t, float):
            t = torch.tensor(t)
        return ((t*self.num_timesteps).round().long()).clip(min=0, max=self.num_timesteps-1)

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        '''
        :param t: 0-1, rate of time step
        '''
        if isinstance(t, float):
            t = torch.tensor(t)
        return ((t*self.num_timesteps).round().long()).clip(min=0, max=self.num_timesteps-1)

    def c_noise(self, t: Union[float, torch.Tensor]):
        return (t*self.num_timesteps).round()