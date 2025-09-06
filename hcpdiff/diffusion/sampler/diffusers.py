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

        self.sigma_scheduler.c_in = self.c_in

    def c_in(self, t):
        one = torch.ones_like(t)
        # if hasattr(self.scheduler, '_step_index'):
        #     self.scheduler._step_index = None
        return self.scheduler.scale_model_input(one, t)

    def get_timesteps(self, N_steps, device='cuda'):
        self.scheduler.set_timesteps(N_steps, device=device)
        return self.scheduler.timesteps / self.sigma_scheduler.num_timesteps # Normalize timesteps to [0, 1]

    def init_noise(self, shape, device='cuda', dtype=torch.float32):
        return randn_tensor(shape, generator=self.generator, device=device, dtype=dtype)*self.scheduler.init_noise_sigma

    def add_noise(self, x, t):
        noise = randn_tensor(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)
        t_in = self.sigma_scheduler.c_noise(t)
        return self.scheduler.add_noise(x, noise, t_in), noise

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

    def denoise(self, x_t, t, eps=None, generator=None):
        t_in = self.sigma_scheduler.c_noise(t)
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator, self.eta)
        return self.scheduler.step(eps, t_in, x_t, **extra_step_kwargs).prev_sample
