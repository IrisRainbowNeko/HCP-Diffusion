import random

import torch
from torch.nn import functional as F

from hcpdiff.diffusion.sampler import BaseSampler

class PyramidNoiseSampler:
    def __init__(self, level: int = 6, discount: float = 0.4, step_size: float = 2., resize_mode: str = 'bilinear'):
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

    @classmethod
    def patch(cls, base_sampler: BaseSampler, level: int = 6, discount: float = 0.4, step_size: float = 2., resize_mode: str = 'bilinear'):
        patcher = cls(level, discount, step_size, resize_mode)
        base_sampler.make_nosie = patcher.make_nosie
        return base_sampler

if __name__ == '__main__':
    from hcpdiff.diffusion.sampler import EDM_DDPMSampler, DDPMContinuousSigmaScheduler
    from matplotlib import pyplot as plt

    sampler = PyramidNoiseSampler.patch(EDM_DDPMSampler(DDPMContinuousSigmaScheduler()))
    noise = sampler.make_nosie((1,3,512,512), device='cpu')
    plt.figure()
    plt.imshow(noise[0].permute(1,2,0))
    plt.show()
