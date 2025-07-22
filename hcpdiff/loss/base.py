from rainbowneko.train.loss import LossContainer
from typing import Dict, Any
from torch import Tensor

class DiffusionLossContainer(LossContainer):
    def __init__(self, loss, weight=1.0, key_map=None):
        key_map = key_map or getattr(loss, '_key_map', None) or ('pred.model_pred -> 0', 'pred.target -> 1')
        super().__init__(loss, weight, key_map)
        self.target_type = getattr(loss, 'target_type', None)

    def get_target(self, model_pred, x_0, noise, x_t, timesteps, noise_sampler, **kwargs):
        # Get target
        target = noise_sampler.get_target(x_0, x_t, timesteps, eps=noise, target_type=self.target_type)

        # Convert pred_type for target_type
        pred = noise_sampler.pred_for_target(model_pred, x_t, timesteps, eps=noise, target_type=self.target_type)
        return pred, target
    
    def forward(self, pred:Dict[str,Any], inputs:Dict[str,Any]) -> Tensor:
        pred_cvt, target = self.get_target(**pred)
        pred['model_pred'] = pred_cvt
        pred['target'] = target
        loss = super().forward(pred, inputs) # [B,*,*,*]
        return loss.mean()