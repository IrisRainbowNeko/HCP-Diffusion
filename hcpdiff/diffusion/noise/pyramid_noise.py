import random

import torch
from torch.nn import functional as F

from .noise_base import NoiseBase

class PyramidNoiseScheduler(NoiseBase):
    def __init__(self, base_scheduler, level: int = 6, discount: float = 0.4, step_size: float = 2., resize_mode: str = 'bilinear'):
        super().__init__(base_scheduler)
        self.level = level
        self.step_size = step_size
        self.resize_mode = resize_mode
        self.discount = discount

    def make_nosie(self, shape, device='cuda', dtype=torch.float32):
        noise = torch.randn(shape, device=device, dtype=dtype)
        with torch.no_grad():
            b, c, h, w = noise.shape
            for i in range(1, self.level):
                r = random.random()*2+self.step_size
                wn, hn = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
                noise += F.interpolate(torch.randn(b, c, hn, wn).to(noise), (h, w), None, self.resize_mode)*(self.discount**i)
                if wn == 1 or hn == 1:
                    break
            noise = noise/noise.std()
        return noise


# if __name__ == '__main__':
#     noise = torch.randn(1,3,512,512)
#     level=10
#     discount=0.6
#     b, c, h, w = noise.shape
#     for i in range(level):
#         r = random.random() * 2 + 2
#         wn, hn = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
#         noise += F.interpolate(torch.randn(b, c, wn, hn).to(noise), (w, h), None, 'bilinear') * discount ** i
#         if wn == 1 or hn == 1:
#             break
#     noise = noise / noise.std()
#
#     from matplotlib import pyplot as plt
#     plt.figure()
#     plt.imshow(noise[0].permute(1,2,0))
#     plt.show()
