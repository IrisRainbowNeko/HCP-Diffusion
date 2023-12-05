import torch
from torch import nn
from copy import deepcopy
from typing import Iterable, Tuple, Dict
import numpy as np

class ModelEMA:
    def __init__(self, model: nn.Module, decay_max=0.9997, inv_gamma=1., power=2/3, start_step=0, device='cpu'):
        self.train_params = {name:p.data.to(device) for name, p in model.named_parameters() if p.requires_grad}
        self.train_params.update({name:p.to(device) for name, p in model.named_buffers()})
        self.decay_max = decay_max
        self.inv_gamma = inv_gamma
        self.power = power
        self.step = start_step
        self.device=device

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        # Compute the decay factor for the exponential moving average.
        decay = 1-(1+self.step/self.inv_gamma)**-self.power
        decay = np.clip(decay, 0., self.decay_max)

        for name, param in model.named_parameters():
            if name in self.train_params:
                self.train_params[name].lerp_(param.data.to(self.device), 1-decay) # (1-e)x + e*x_

        for name, param in model.named_buffers():
            if name in self.train_params:
                self.train_params[name].copy_(param.to(self.device))

        #torch.cuda.empty_cache()

    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.train_params:
                param.data.copy_(self.train_params[name])

    def to(self, device=None, dtype=None):
        # .to() on the tensors handles None correctly
        self.train_params = {
            name:(p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)) for name, p in self.train_params.items()
        }
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.train_params

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        for k, v in state:
            if k in self.train_params:
                self.train_params[k]=v
