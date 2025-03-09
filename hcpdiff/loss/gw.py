import torch
from torch import nn
from torch.nn import functional as F

class GWLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.sobel_x = torch.FloatTensor(sobel_x)
        self.sobel_y = torch.FloatTensor(sobel_y)
        self.register_buffer('sobel_x', self.sobel_x)
        self.register_buffer('sobel_y', self.sobel_y)

    def forward(self, pred, target):
        '''

        :param pred: [B,C,H,W]
        :param target: [B,C,H,W]
        :return: [B,C,H,W]
        '''
        b, c, w, h = pred.shape

        sobel_x = self.sobel_x.expand(c, 1, 3, 3).to(pred.device)
        sobel_y = self.sobel_y.expand(c, 1, 3, 3).to(pred.device)
        Ix1 = F.conv2d(pred, sobel_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(target, sobel_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(pred, sobel_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(target, sobel_y, stride=1, padding=1, groups=c)

        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        loss = (1 + 4 * dx) * (1 + 4 * dy) * torch.abs(pred - target)
        return loss