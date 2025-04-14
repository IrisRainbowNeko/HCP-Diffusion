from typing import Tuple
import torch
from .sigma_scheduler import SigmaScheduler
from diffusers import DDPMScheduler

class BaseSampler:
    def __init__(self, sigma_scheduler: SigmaScheduler, generator: torch.Generator = None):
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
        return getattr(self.sigma_scheduler, 'num_timesteps', 1000.)

    def get_timesteps(self, N_steps, device='cuda'):
        return torch.linspace(0, self.num_timesteps, N_steps, device=device)

    def make_nosie(self, shape, device='cuda', dtype=torch.float32):
        return torch.randn(shape, generator=self.generator, device=device, dtype=dtype)

    def init_noise(self, shape, device='cuda', dtype=torch.float32):
        sigma = self.sigma_scheduler.sigma_max
        return self.make_nosie(shape, device, dtype)*sigma

    def add_noise(self, x, sigma) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = self.make_nosie(x.shape, device=x.device)
        noisy_x = (x.to(dtype=torch.float32)-self.c_out(sigma)*noise)/self.c_skip(sigma)
        return noisy_x.to(dtype=x.dtype), noise.to(dtype=x.dtype)

    def add_noise_rand_t(self, x):
        bs = x.shape[0]
        # timesteps: [0, 1]
        sigma, timesteps = self.sigma_scheduler.sample_sigma(shape=(bs,))
        sigma = sigma.view(-1, 1, 1, 1).to(x.device)
        timesteps = timesteps.to(x.device)
        noisy_x, noise = self.add_noise(x, sigma)

        # Sample a random timestep for each image
        timesteps = timesteps*(self.num_timesteps-1)
        return noisy_x, noise, sigma, timesteps

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError

    def eps_to_x0(self, eps, x_t, sigma):
        return self.c_skip(sigma)*x_t+self.c_out(sigma)*eps

    def velocity_to_eps(self, v_pred, x_t, sigma):
        alpha = 1/(sigma**2+1)
        sqrt_alpha = alpha.sqrt()
        one_sqrt_alpha = (1-alpha).sqrt()
        return sqrt_alpha*v_pred + one_sqrt_alpha*(x_t*sqrt_alpha)

    def eps_to_velocity(self, eps, x_t, sigma):
        alpha = 1/(sigma**2+1)
        sqrt_alpha = alpha.sqrt()
        one_sqrt_alpha = (1-alpha).sqrt()
        return eps/sqrt_alpha - one_sqrt_alpha*x_t

    def velocity_to_x0(self, v_pred, x_t, sigma):
        alpha = 1/(sigma**2+1)
        one_sqrt_alpha = (1-alpha).sqrt()
        return alpha*x_t - one_sqrt_alpha*v_pred
