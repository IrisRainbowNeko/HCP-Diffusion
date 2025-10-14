import torch
from torch import nn
from torch.nn import functional as F

class GWLoss(nn.Module):
    def __init__(self, eps=1e-3, size_average=True):
        super().__init__()

        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        sobel_x = torch.FloatTensor(sobel_x)
        sobel_y = torch.FloatTensor(sobel_y)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.eps = eps
        self.size_average = size_average

    def forward(self, pred, target):
        '''

        :param pred: [B,C,H,W]
        :param target: [B,C,H,W]
        :return: [B,C,H,W]
        '''
        b, c, w, h = pred.shape

        target = target.to(dtype=torch.float32)
        pred = pred.to(dtype=torch.float32)
        sobel_x = self.sobel_x.expand(c, 1, 3, 3).to(pred.device, dtype=torch.float32)
        sobel_y = self.sobel_y.expand(c, 1, 3, 3).to(pred.device, dtype=torch.float32)
        Ix1 = F.conv2d(pred, sobel_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(target, sobel_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(pred, sobel_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(target, sobel_y, stride=1, padding=1, groups=c)

        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        diff = pred - target
        loss = torch.sqrt((diff*diff)+(self.eps*self.eps))
        loss = (1 + 4 * dx) * (1 + 4 * dy) * loss
        if self.size_average:
            loss = loss.mean()
        return loss