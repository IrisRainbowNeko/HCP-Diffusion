from torch.nn.modules.loss import _Loss
import torch

def edge_loss(y_pred, y_true):
    dy_pred, dx_pred = torch.gradient(y_pred)
    dy_true, dx_true = torch.gradient(y_true)
    return torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))


class SSIMLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dy_pred, dx_pred = torch.gradient(input)
        dy_true, dx_true = torch.gradient(target)
        return 1-self.ssim(input, target).view(-1,1,1,1)