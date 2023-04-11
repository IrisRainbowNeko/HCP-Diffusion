import torch
from einops import repeat
import math

class CFGContext:
    def pre(self, noisy_latents, timesteps):
        return noisy_latents, timesteps

    def post(self, model_pred):
        return model_pred

class DreamArtistPTContext(CFGContext):
    def __init__(self, cfg_scale, num_train_timesteps):
        self.cfg_scale=cfg_scale
        self.num_train_timesteps=num_train_timesteps

    def pre(self, noisy_latents, timesteps):
        self.t_raw = timesteps
        noisy_latents = repeat(noisy_latents, 'b c h w -> (pn b) c h w', pn=2)
        timesteps = timesteps.repeat(2)
        return noisy_latents, timesteps

    def post(self, model_pred):
        e_t_uncond, e_t = model_pred.chunk(2)
        if self.cfg_scale[0] != self.cfg_scale[1]:
            rate = self.t_raw / (self.num_train_timesteps - 1)
            if self.cfg_scale[2] == 'cos':
                rate = torch.cos((rate - 1) * math.pi / 2)
            elif self.cfg_scale[2] == 'cos2':
                rate = 1 - torch.cos(rate * math.pi / 2)
            elif self.cfg_scale[2] == 'ln':
                pass
            else:
                rate = eval(self.cfg_scale[2])
            rate = rate.view(-1,1,1,1)
        else:
            rate = 1
        model_pred = e_t_uncond + ((self.cfg_scale[1] - self.cfg_scale[0]) * rate + self.cfg_scale[0]) * (e_t - e_t_uncond)
        return model_pred