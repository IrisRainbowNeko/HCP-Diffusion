from torch import nn

from .base import DiffusionLossContainer

class LossWeight(nn.Module):
    def __init__(self, loss: DiffusionLossContainer):
        super().__init__()
        self.loss = loss

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
        if self.loss.target_type == 'eps':
            return 1
        elif self.loss.target_type == "x0":
            sigma = pred['sigma']
            return (1./sigma**2).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"{self.__class__.__name__} is not support for target_type {self.loss.target_type}")

class MinSNRWeight(LossWeight):
    def __init__(self, loss: DiffusionLossContainer, gamma: float = 1.):
        super().__init__(loss)
        self.gamma = gamma

    def get_weight(self, pred, inputs):
        sigma = pred['sigma']
        if self.loss.target_type == 'eps':
            w_snr = (self.gamma*sigma**2).clip(max=1).float()
        elif self.loss.target_type == "x0":
            w_snr = (1/(sigma**2)).clip(max=self.gamma).float()
        else:
            raise ValueError(f"{self.__class__.__name__} is not support for target_type {self.loss.target_type}")

        return w_snr.view(-1, 1, 1, 1)

class EDMWeight(LossWeight):
    def __init__(self, loss: DiffusionLossContainer, gamma: float = 1.):
        super().__init__(loss)
        self.gamma = gamma

    def get_weight(self, pred, inputs):
        sigma = pred['sigma']
        if self.loss.target_type == 'eps':
            w_snr = ((sigma**2+self.gamma**2)/(self.gamma**2)).float()
        elif self.loss.target_type == "x0":
            w_snr = ((sigma**2+self.gamma**2)/((sigma*self.gamma)**2)).float()
        else:
            raise ValueError(f"{self.__class__.__name__} is not support for target_type {self.loss.target_type}")

        return w_snr.view(-1, 1, 1, 1)
