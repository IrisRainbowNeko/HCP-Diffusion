import torch
import math
from typing import Union, Tuple
from hcpdiff.utils import linear_interp
from .base import SigmaScheduler

class DDPMDiscreteSigmaScheduler(SigmaScheduler):
    def __init__(self, beta_schedule: str = "scaled_linear", linear_start=0.00085, linear_end=0.0120, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = self.make_betas(beta_schedule, linear_start, linear_end, num_timesteps)
        alphas = 1.0-self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sigmas = ((1-self.alphas_cumprod)/self.alphas_cumprod).sqrt()

        # for VLB calculation
        self.alphas_cumprod_prev = torch.cat([alphas.new_tensor([1.0]), self.alphas_cumprod[:-1]])
        self.posterior_mean_coef1 = self.betas*torch.sqrt(self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0-self.alphas_cumprod_prev)*torch.sqrt(alphas)/(1.0-self.alphas_cumprod)

        self.posterior_variance = self.betas*(1.0-self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))


    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_sigma(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor(t)
        return self.sigmas[(t*len(self.sigmas)).long()]

    def sample_sigma(self, min_rate=0.0, max_rate=1.0, shape=(1,)):
        if isinstance(min_rate, float):
            min_rate = torch.full(shape, min_rate)
        if isinstance(max_rate, float):
            max_rate = torch.full(shape, max_rate)

        t = torch.lerp(min_rate, max_rate, torch.rand_like(min_rate))
        t_scale = (t*(self.num_timesteps-1e-5)).long()  # [0, num_timesteps-1)
        return self.sigmas[t_scale], t

    def sigma_to_t(self, sigma: Union[float, torch.Tensor]):
        t = (self.sigmas-sigma).abs().argmin()
        return t/self.num_timesteps

    def get_post_mean(self, t, x_0, x_t):
        t = (t*len(self.sigmas)).long()
        return self.posterior_mean_coef1[t].view(-1, 1, 1, 1).to(t.device)*x_0 + self.posterior_mean_coef2[t].view(-1, 1, 1, 1).to(t.device)*x_t

    def get_post_log_var(self, t, x_t_var=None):
        t = (t*len(self.sigmas)).long()
        min_log = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1).to(t.device)
        if x_t_var is None:
            return min_log
        else:
            max_log = self.betas.log()[t].view(-1, 1, 1, 1).to(t.device)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (x_t_var+1)/2
            model_log_variance = frac*max_log+(1-frac)*min_log
            return model_log_variance


    @staticmethod
    def betas_for_alpha_bar(
        num_diffusion_timesteps,
        max_beta=0.999,
        alpha_transform_type="cosine",
    ):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
        (1-beta) over time from t = [0,1].

        Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
        to that part of the diffusion process.


        Args:
            num_diffusion_timesteps (`int`): the number of betas to produce.
            max_beta (`float`): the maximum beta to use; use values lower than 1 to
                         prevent singularities.
            alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                         Choose from `cosine` or `exp`

        Returns:
            betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
        """
        if alpha_transform_type == "cosine":

            def alpha_bar_fn(t):
                return math.cos((t+0.008)/1.008*math.pi/2)**2

        elif alpha_transform_type == "exp":

            def alpha_bar_fn(t):
                return math.exp(t*-12.0)

        else:
            raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i/num_diffusion_timesteps
            t2 = (i+1)/num_diffusion_timesteps
            betas.append(min(1-alpha_bar_fn(t2)/alpha_bar_fn(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    @staticmethod
    def make_betas(beta_schedule, beta_start, beta_end, num_train_timesteps, betas=None):
        if betas is not None:
            return torch.tensor(betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            return torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32)**2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            return DDPMDiscreteSigmaScheduler.betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            return torch.sigmoid(betas)*(beta_end-beta_start)+beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented.")

class DDPMContinuousSigmaScheduler(DDPMDiscreteSigmaScheduler):

    def get_sigma(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor(t)
        return linear_interp(self.sigmas, t)

    def sample_sigma(self, min_rate=0.0, max_rate=1.0, shape=(1,)):
        if isinstance(min_rate, float):
            min_rate = torch.full(shape, min_rate)
        if isinstance(max_rate, float):
            max_rate = torch.full(shape, max_rate)

        t = torch.lerp(min_rate, max_rate, torch.rand_like(min_rate))
        t_scale = (t*(self.num_timesteps-1-1e-5))  # [0, num_timesteps-1)

        return linear_interp(self.sigmas, t_scale), t

    def sigma_to_t(self, sigma: Union[float, torch.Tensor]):
        diff = self.sigmas-sigma
        diff[diff<0] = float('inf')
        t0 = diff.argmin().clamp(0, self.num_timesteps-2)
        return t0 + diff.min()/(self.sigmas[t0+1]-self.sigmas[t0])

class TimeSigmaScheduler(SigmaScheduler):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps

    def get_sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        '''
        :param t: 0-1, rate of time step
        '''
        return t

    def sample_sigma(self, min_rate=0.0, max_rate=1.0, shape=(1,)) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(min_rate, float):
            min_rate = torch.full(shape, min_rate)
        if isinstance(max_rate, float):
            max_rate = torch.full(shape, max_rate)

        t = torch.lerp(min_rate, max_rate, torch.rand_like(min_rate))
        t_scale = (t*(self.num_timesteps-1e-5)).long()  # [0, num_timesteps-1)
        return t_scale, t

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np

    sigma_scheduler = DDPMDiscreteSigmaScheduler()
    print(sigma_scheduler.sigma_min, sigma_scheduler.sigma_max)
    t = torch.linspace(0, 1, 1000)
    rho = 1.
    s2 = (sigma_scheduler.sigma_min**(1/rho)+t*(sigma_scheduler.sigma_max**(1/rho)-sigma_scheduler.sigma_min**(1/rho)))**rho
    t2 = np.interp(s2.log().numpy(), sigma_scheduler.sigmas.log().numpy(), t.numpy())

    plt.figure()
    plt.plot(sigma_scheduler.sigmas)
    plt.plot(t2*1000, s2)
    plt.show()

    plt.figure()
    plt.plot(sigma_scheduler.sigmas.log())
    plt.plot(t2*1000, s2.log())
    plt.show()
