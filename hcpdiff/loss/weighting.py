from torch import nn

from .base import DiffusionLossContainer

class LossWeight(nn.Module):
    def __init__(self, loss: DiffusionLossContainer):
        super().__init__()
        self.loss = loss

    def get_c_out(self, pred):
        t = pred['timesteps']
        noise_sampler = pred['noise_sampler']
        return noise_sampler.sigma_scheduler.c_out(t)

    def get_weight(self, pred, inputs):
        '''

        :param input: [B,C,H,W]
        :param target: [B,C,H,W]
        :return: [B,1,1,1] or [B,C,H,W]
        '''
        raise NotImplementedError

    def forward(self, pred, inputs):
        '''
        weight: [B,1,1,1] or [B,C,H,W]
        loss: [B,*,*,*]
        '''
        return self.get_weight(pred, inputs)*self.loss(pred, inputs)

class SNRWeight(LossWeight):
    def get_weight(self, pred, inputs):
        noise_sampler = pred['noise_sampler']
        c_out = self.get_c_out(pred)
        target_type = getattr(self.loss, 'target_type', None) or noise_sampler.target_type
        if target_type == 'eps':
            w_snr = 1
        elif target_type == "x0":
            w_snr = (1./c_out**2).float()
        elif target_type == "velocity":
            w_snr = (1./(1-c_out)**2).float()
        else:
            raise ValueError(f"{self.__class__.__name__} is not support for target_type {target_type}")

        return w_snr.view(-1, 1, 1, 1)

class MinSNRWeight(LossWeight):
    def __init__(self, loss: DiffusionLossContainer, gamma: float = 1.):
        super().__init__(loss)
        self.gamma = gamma

    def get_weight(self, pred, inputs):
        noise_sampler = pred['noise_sampler']
        c_out = self.get_c_out(pred)
        target_type = getattr(self.loss, 'target_type', None) or noise_sampler.target_type
        if target_type == 'eps':
            w_snr = (self.gamma*c_out**2).clip(max=1).float()
        elif target_type == "x0":
            w_snr = (1./c_out**2).clip(max=self.gamma).float()
        elif target_type == "velocity":
            w_v = 1/(1-c_out)**2
            w_snr = (self.gamma*c_out**2/w_v).clip(max=w_v).float()
        else:
            raise ValueError(f"{self.__class__.__name__} is not support for target_type {target_type}")

        return w_snr.view(-1, 1, 1, 1)

class EDMWeight(LossWeight):
    def __init__(self, loss: DiffusionLossContainer, gamma: float = 1.):
        super().__init__(loss)
        self.gamma = gamma

    def get_weight(self, pred, inputs):
        c_out = self.get_c_out(pred)
        noise_sampler = pred['noise_sampler']
        target_type = getattr(self.loss, 'target_type', None) or noise_sampler.target_type
        if target_type == 'edm':
            w_snr = 1
        elif target_type == "x0":
            w_snr = (1./c_out**2).float()
        else:
            raise ValueError(f"{self.__class__.__name__} is not support for target_type {target_type}")

        return w_snr.view(-1, 1, 1, 1)
