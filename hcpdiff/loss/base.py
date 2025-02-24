from rainbowneko.train.loss import LossContainer

class DiffusionLossContainer(LossContainer):
    def __init__(self, loss, weight=1.0, key_map=None):
        key_map = key_map or getattr(loss, '_key_map', None) or ('pred.model_pred -> 0', 'pred.target -> 1')
        super().__init__(loss, weight, key_map)
        self.target_type = getattr(loss, 'target_type', 'eps')

    def get_target(self, pred_type, model_pred, x_0, noise, x_t, sigma, noise_sampler, **kwargs):
        # Get target
        if self.target_type == "eps":
            target = noise
        elif self.target_type == "x0":
            target = x_0
        elif self.target_type == "velocity":
            target = noise_sampler.eps_to_velocity(noise, x_t, sigma)
        else:
            raise ValueError(f"Unsupport target_type {self.target_type}")

        # # remove pred vars
        # if model_pred.shape[1] == target.shape[1]*2:
        #     model_pred, _ = model_pred.chunk(2, dim=1)

        # Convert pred_type to target_type
        if pred_type != self.target_type:
            cvt_func = getattr(noise_sampler, f'{pred_type}_to_{self.target_type}', None)
            if cvt_func is None:
                raise ValueError(f"Unsupport pred_type {pred_type} with target_type {self.target_type}")
            else:
                model_pred = cvt_func(model_pred, x_t, sigma)
        return model_pred, target
    
    def forward(self, pred, inputs):
        model_pred, target = self.get_target(**pred)
        pred['model_pred'] = model_pred
        pred['target'] = target
        return super().forward(pred, inputs)