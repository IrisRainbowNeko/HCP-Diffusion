from typing import Union, Tuple

import torch

class SigmaScheduler:
    def scale_t(self, t):
        return t

    def sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        '''
        x(t) = \alpha(t)*x(0) + \sigma(t)*eps
        :param t: 0-1, rate of time step
        '''
        raise NotImplementedError

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        '''
        x(t) = \alpha(t)*x(0) + \sigma(t)*eps
        :param t: 0-1, rate of time step
        '''
        raise NotImplementedError

    def velocity(self, t: Union[float, torch.Tensor], dt=1e-8, normlize=True) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        v(t) = dx(t)/dt = d\alpha(t)/dt * x(0) + d\sigma(t)/dt *eps
        :param t: 0-1, rate of time step
        :return: d\alpha(t)/dt, d\sigma(t)/dt
        '''
        d_alpha = (self.alpha(t+dt)-self.alpha(t))/dt
        d_sigma = (self.sigma(t+dt)-self.sigma(t))/dt
        if normlize:
            norm = torch.sqrt(d_alpha**2+d_sigma**2)
            return d_alpha/norm, d_sigma/norm
        else:
            return d_alpha, d_sigma

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
        raise NotImplementedError

    def sigma_to_alpha(self, sigma):
        raise NotImplementedError

    def c_in(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            return 1.
        else:
            return torch.ones_like(t, dtype=torch.float32)

    def c_skip(self, t: Union[float, torch.Tensor]):
        '''
        \hat{x}(0) = c_skip*x(t) + c_out*f(x(t))
        :param t: 0-1, rate of time step
        '''
        return 1./self.alpha(t)

    def c_out(self, t: Union[float, torch.Tensor]):
        '''
        \hat{x}(0) = c_skip*x(t) + c_out*f(x(t))
        :param t: 0-1, rate of time step
        '''
        return -self.sigma(t)/self.alpha(t)
    
    def c_noise(self, t: Union[float, torch.Tensor]):
        return t

    def sample(self, min_t=0.0, max_t=1.0, shape=(1,)) -> torch.Tensor:
        if isinstance(min_t, float):
            min_t = torch.full(shape, min_t)
        if isinstance(max_t, float):
            max_t = torch.full(shape, max_t)

        t = torch.lerp(min_t, max_t, torch.rand_like(min_t))
        return t
