import math
from typing import Union, Callable

import torch
from einops import repeat
from rainbowneko.utils import add_dims

class CFGContext:
    def pre(self, noisy_latents, timesteps):
        return noisy_latents, timesteps

    def post(self, model_pred):
        return model_pred

class DreamArtistPTContext(CFGContext):
    def __init__(self, cfg_low: float, cfg_high: float=None, cfg_func: Union[str, Callable]=None, num_train_timesteps=1000):
        self.cfg_low = cfg_low
        self.cfg_high = cfg_high or cfg_low
        self.cfg_func = cfg_func
        self.num_train_timesteps = num_train_timesteps

    def pre(self, noisy_latents, timesteps):
        self.t_raw = timesteps
        noisy_latents = repeat(noisy_latents, 'b c h w -> (pn b) c h w', pn=2)
        timesteps = timesteps.repeat(2)
        return noisy_latents, timesteps

    def post(self, model_pred):
        e_t_uncond, e_t = model_pred.chunk(2)
        if self.cfg_low != self.cfg_high:
            rate = self.t_raw/(self.num_train_timesteps-1)
            if self.cfg_func == 'cos':
                rate = torch.cos((rate-1)*math.pi/2)
            elif self.cfg_func == 'cos2':
                rate = 1-torch.cos(rate*math.pi/2)
            elif self.cfg_func == 'ln':
                pass
            else:
                rate = self.cfg_func(rate)
            rate = add_dims(rate, model_pred.ndim-1)
        else:
            rate = 1
        model_pred = e_t_uncond+((self.cfg_high-self.cfg_low)*rate+self.cfg_low)*(e_t-e_t_uncond)
        return model_pred
