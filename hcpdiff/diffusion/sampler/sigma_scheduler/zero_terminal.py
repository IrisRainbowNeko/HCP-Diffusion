from typing import Union

import torch
from .base import SigmaScheduler

class ZeroTerminalScheduler(SigmaScheduler):
    def __init__(self, ref_scheduler: SigmaScheduler, eps=1e-4):
        self.ref_scheduler = ref_scheduler
        self.eps = eps

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        alpha_0 = self.ref_scheduler.alpha_start
        alpha_T = self.ref_scheduler.alpha_end
        alpha = self.ref_scheduler.alpha(t)
        return (alpha - alpha_T)*(alpha_0-self.eps)/(alpha_0 - alpha_T) + self.eps

    def sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        try:
            alpha = self.alpha(t)
            return self.ref_scheduler.alpha_to_sigma(alpha)
        except NotImplementedError:
            raise NotImplementedError(f'{type(self.ref_scheduler)} cannot be a "ZeroTerminalScheduler"!')
