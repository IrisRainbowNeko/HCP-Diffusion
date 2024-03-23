import torch
from .sigma_scheduler import SigmaScheduler

class BaseSampler:
    def __init__(self, sigma_scheduler: SigmaScheduler, generator: torch.Generator=None):
        self.sigma_scheduler = sigma_scheduler
        self.generator = generator

    def c_in(self, sigma):
        return 1

    def c_out(self, sigma):
        return 1

    def c_skip(self, sigma):
        return 1

    @property
    def num_timesteps(self):
        return getattr(self.sigma_scheduler, 'num_timesteps', 1.)

    def make_nosie(self, shape, device='cuda', dtype=torch.float32):
        return torch.randn(shape, generator=self.generator, device=device, dtype=dtype)

    def init_noise(self, shape, device='cuda', dtype=torch.float32):
        sigma = self.sigma_scheduler.sigma_max
        return self.make_nosie(shape, device, dtype)*sigma

    def add_noise(self, x, sigma):
        raise NotImplementedError

    def add_noise_rand_t(self, x):
        bs = x.shape[0]
        # timesteps: [0, 1]
        sigma, timesteps = self.sigma_scheduler.sample_sigma(shape=(bs,))
        sigma = sigma.view(-1,1,1,1).to(x.device)
        timesteps = timesteps.to(x.device)
        noisy_x = self.add_noise(x, sigma).to(dtype=x.dtype)

        # Sample a random timestep for each image
        timesteps = timesteps*(self.num_timesteps-1)
        return noisy_x, sigma, timesteps

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError

    def get_x0(self, eps, x_t, sigma):
        return self.c_skip(sigma)*x_t+self.c_out(sigma)*eps