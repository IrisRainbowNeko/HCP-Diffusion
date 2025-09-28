import torch

class Timer:
    def sample(self, shape=(1,)) -> torch.Tensor:
        return torch.rand(shape)

class LogitNormalTimer(Timer):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(self, shape=(1,)) -> torch.Tensor:
        return torch.sigmoid(torch.normal(mean=self.mean, std=self.std, size=shape))
