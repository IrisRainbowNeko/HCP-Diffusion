from pytorch_msssim import SSIM, MS_SSIM
from torch.nn.modules.loss import _Loss
import torch

class SSIMLoss(_Loss):
    target_type = 'x0'

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.ssim = SSIM(data_range=1., size_average=False, channel=4)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''

        :param input: [B,C,H,W]
        :param target: [B,C,H,W]
        :return: [B,1,1,1]
        '''
        input = (input+1)/2
        target = (target+1)/2
        return 1-self.ssim(input, target).view(-1,1,1,1)

class MS_SSIMLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.ssim = MS_SSIM(data_range=1., size_average=False, channel=4)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''

        :param input: [B,C,H,W]
        :param target: [B,C,H,W]
        :return: [B,1,1,1]
        '''
        input = (input+1)/2
        target = (target+1)/2
        return 1-self.ssim(input, target).view(-1,1,1,1)