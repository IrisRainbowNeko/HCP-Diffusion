import torch
import inspect
from diffusers import SchedulerMixin, DDPMScheduler

try:
    from diffusers.utils import randn_tensor
except:
    # new version of diffusers
    from diffusers.utils.torch_utils import randn_tensor

from .base import BaseSampler
from .sigma_scheduler import TimeSigmaScheduler

class DiffusersSampler(BaseSampler):
    def __init__(self, scheduler: SchedulerMixin, eta=0.0, generator: torch.Generator=None):
        sigma_scheduler = TimeSigmaScheduler()
        super().__init__(sigma_scheduler, generator)
        self.scheduler = scheduler
        self.eta = eta

    def c_in(self, sigma):
        one = torch.ones_like(sigma)
        if hasattr(self.scheduler, '_step_index'):
            self.scheduler._step_index = None
        return self.scheduler.scale_model_input(one, sigma)

    def c_out(self, sigma):
        return -sigma

    def c_skip(self, sigma):
        if self.c_in(sigma) == 1.:  # DDPM model
            return (sigma**2+1).sqrt()  # 1/sqrt(alpha_)
        else:  # EDM model
            return 1.

    def get_timesteps(self, N_steps, device='cuda'):
        self.scheduler.set_timesteps(N_steps, device=device)
        return self.scheduler.timesteps

    def init_noise(self, shape, device='cuda', dtype=torch.float32):
        return randn_tensor(shape, generator=self.generator, device=device, dtype=dtype)*self.scheduler.init_noise_sigma

    def add_noise(self, x, sigma):
        noise = randn_tensor(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)
        return self.scheduler.add_noise(x, noise, sigma), noise

    def prepare_extra_step_kwargs(self, scheduler, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def denoise(self, x_t, sigma, eps=None, generator=None):
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator, self.eta)
        return self.scheduler.step(eps, sigma, x_t, **extra_step_kwargs).prev_sample
