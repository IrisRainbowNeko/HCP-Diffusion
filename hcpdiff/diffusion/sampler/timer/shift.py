import torch
import math
from torch import Tensor

from .base import TimeSampler

class ShiftTimeSampler(TimeSampler):
    def __init__(self, t_sampler: TimeSampler = None, base_reso=1024*1024):
        self.t_sampler = t_sampler
        self.base_reso = base_reso

    def sample(self, min_t=0.0, max_t=1.0, shape=(1,), reso=0) -> torch.Tensor:
        t = self.t_sampler.sample(min_t, max_t, shape)
        shift = math.sqrt(self.base_reso/(reso))
        t = (t*shift)/(1+(shift-1)*t)
        return t

class FluxShiftTimeSampler(TimeSampler):
    def __init__(self, t_sampler: TimeSampler = None, base_shift: float = 0.5, max_shift: float = 1.15, base_reso=256, max_reso=4096):
        self.t_sampler = t_sampler
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_reso = base_reso
        self.max_reso = max_reso

    def time_shift(self, mu: float|Tensor, sigma: float, t: Tensor):
        if torch.is_tensor(mu):
            mu = mu.to(t.device)
            return torch.exp(mu)/(torch.exp(mu)+(1/t-1)**sigma)
        else:
            return math.exp(mu)/(math.exp(mu)+(1/t-1)**sigma)

    def get_lin_function(self, xi, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
        '''
        ^
        |      .(x2,y2)
        |     /
        |   . (x1,y1)
        |_________>
        '''
        m = (y2-y1)/(x2-x1)
        b = y1-m*x1
        return m*xi+b

    def sample(self, min_t=0.0, max_t=1.0, shape=(1,), reso=0) -> torch.Tensor:
        mu = self.get_lin_function(reso, x1=self.base_reso, y1=self.base_shift, x2=self.max_reso, y2=self.max_shift)
        t = self.t_sampler.sample(min_t, max_t, shape)
        t = self.time_shift(mu, 1.0, t)
        return t
