import torch
from torch import nn
from copy import deepcopy
from typing import Iterable, Tuple, Dict

'''class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)'''

class ModelEMA:
    def __init__(self, parameters: Iterable[Tuple[str, torch.nn.Parameter]], decay_max=0.997, decay_factor=(1,5), optimization_step=0):
        self.train_params = {name:p.data.clone().detach() for name, p in parameters}
        self.decay_max = decay_max
        self.decay_factor = decay_factor
        self.optimization_step = optimization_step

    @torch.no_grad()
    def step(self, parameters: Iterable[Tuple[str, torch.nn.Parameter]]):
        self.optimization_step += 1
        # Compute the decay factor for the exponential moving average.
        if self.decay_factor[0]==self.decay_factor[1]:
            one_minus_decay = 1-self.decay_max
        else:
            value = (self.decay_factor[0] + self.optimization_step) / (self.decay_factor[1] + self.optimization_step)
            one_minus_decay = 1 - min(self.decay_max, value)

        for name, param in parameters:
            if name in self.train_params:
                s_param=self.train_params[name]
                s_param.sub_(one_minus_decay * (s_param - param.data.to(s_param.device)))

        #torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> None:
        for name, param in parameters:
            if name in self.train_params:
                param.data.copy_(self.train_params[name])

    def to(self, device=None, dtype=None):
        # .to() on the tensors handles None correctly
        self.train_params = {
            name:(p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)) for name, p in self.train_params
        }
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.train_params

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        for k, v in state:
            if k in self.train_params:
                self.train_params[k]=v
