import torch

class TimeSampler:
    def sample(self, min_t=0.0, max_t=1.0, shape=(1,), reso=0) -> torch.Tensor:
        if isinstance(min_t, float):
            min_t = torch.full(shape, min_t)
        if isinstance(max_t, float):
            max_t = torch.full(shape, max_t)

        t = torch.lerp(min_t, max_t, torch.rand_like(min_t))
        return t

class LogitNormalSampler(TimeSampler):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def sample(self, min_t=0.0, max_t=1.0, shape=(1,), reso=0) -> torch.Tensor:
        if isinstance(min_t, float):
            min_t = torch.full(shape, min_t)
        if isinstance(max_t, float):
            max_t = torch.full(shape, max_t)

        t = torch.sigmoid(torch.normal(mean=self.mean, std=self.std, size=shape))
        t = torch.lerp(min_t, max_t, t)
        return t
