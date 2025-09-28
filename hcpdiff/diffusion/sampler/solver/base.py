from typing import Optional, Union, List
import torch

class ODESolver:
    def __init__(self, sample_steps:int=20, v_mean=True):
        self.sample_steps = sample_steps
        self.v_mean = v_mean

    def get_dt(self, scheduler):
        if self.v_mean:
            return 1./self.sample_steps
        else:
            return scheduler.min_dt

    def step(self, x_t, v_t, dt, **kwargs):
        raise NotImplementedError