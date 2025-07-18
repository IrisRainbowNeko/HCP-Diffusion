import math
from typing import Union, Tuple, Callable

import torch
from hcpdiff.utils import invert_func
from rainbowneko.utils import add_dims

from .base import SigmaScheduler

class DDPMDiscreteSigmaScheduler(SigmaScheduler):
    def __init__(self, beta_schedule: str = "scaled_linear", linear_start=0.00085, linear_end=0.0120, num_timesteps=1000, pred_type='eps'):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = self.make_betas(beta_schedule, linear_start, linear_end, num_timesteps)
        alphas = 1.0-self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.alphas = self.alphas_cumprod.sqrt()
        self.sigmas = (1-self.alphas_cumprod).sqrt()
        self.pred_type = pred_type

        # for VLB calculation
        self.alphas_cumprod_prev = torch.cat([alphas.new_tensor([1.0]), self.alphas_cumprod[:-1]])
        self.posterior_mean_coef1 = self.betas*torch.sqrt(self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0-self.alphas_cumprod_prev)*torch.sqrt(alphas)/(1.0-self.alphas_cumprod)

        self.posterior_variance = self.betas*(1.0-self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))

    # def scale_t(self, t):
    #     return t*(self.num_timesteps-1)

    @property
    def sigma_start(self):
        return self.sigmas[0]

    @property
    def sigma_end(self):
        return self.sigmas[-1]

    @property
    def alpha_start(self):
        return self.alphas[0]

    @property
    def alpha_end(self):
        return self.alphas[-1]

    def sigma(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor(t)
        self.sigmas = self.sigmas.to(t.device)
        return self.sigmas[((t*self.num_timesteps).round().long()).clip(min=0, max=self.num_timesteps-1)]

    def alpha(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor(t)
        self.alphas = self.alphas.to(t.device)
        return self.alphas[((t*self.num_timesteps).round().long()).clip(min=0, max=self.num_timesteps-1)]
    
    def c_noise(self, t: Union[float, torch.Tensor]):
        return (t*self.num_timesteps).round()

    def velocity(self, t: Union[float, torch.Tensor], dt=1e-8, normlize=True) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        v(t) = dx(t)/dt = d\alpha(t)/dt * x(0) + d\sigma(t)/dt *eps
        :param t: 0-1, rate of time step
        :return: d\alpha(t)/dt, d\sigma(t)/dt
        '''
        d_alpha = -self.sigma(t)
        d_sigma = self.alpha(t)
        if normlize:
            norm = torch.sqrt(d_alpha**2+d_sigma**2)
            return d_alpha/norm, d_sigma/norm
        else:
            return d_alpha, d_sigma

    def sigma_to_t(self, sigma: Union[float, torch.Tensor]):
        ref_t = np.linspace(0, 1, len(self.sigmas))
        t = torch.tensor(np.interp(sigma.cpu().clip(min=1e-8).log().numpy(), self.sigmas, ref_t))
        return t

    def alpha_to_t(self, alpha: Union[float, torch.Tensor]):
        ref_t = np.linspace(0, 1, len(self.alphas))
        t = torch.tensor(np.interp(alpha.cpu().clip(min=1e-8).log().numpy(), self.alphas, ref_t))
        return t

    def alpha_to_sigma(self, alpha):
        return torch.sqrt(1 - alpha**2)

    def sigma_to_alpha(self, sigma):
        return torch.sqrt(1 - sigma**2)

    def get_post_mean(self, t, x_0, x_t):
        t = (t*len(self.sigmas)).long()
        return add_dims(self.posterior_mean_coef1[t].to(t.device), x_0.ndim-1)*x_0+add_dims(self.posterior_mean_coef2[t].to(t.device), x_t.ndim-1)*x_t

    def get_post_log_var(self, t, ndim, x_t_var=None):
        t = (t*len(self.sigmas)).long()
        min_log = add_dims(self.posterior_log_variance_clipped[t].to(t.device), ndim-1)
        if x_t_var is None:
            return min_log
        else:
            max_log = add_dims(self.betas.log()[t].to(t.device), ndim-1)
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

class DDPMContinuousSigmaScheduler(SigmaScheduler):
    def __init__(self, beta_schedule: str = "scaled_linear", linear_start=0.00085, linear_end=0.0120, t_base=1000):
        self.alpha_bar_fn = self.make_alpha_bar_fn(beta_schedule, linear_start, linear_end)
        self.t_base = t_base  # base time step for continuous product

    def continuous_product(self, alpha_fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor):
        '''

        :param alpha_fn: alpha function
        :param t: timesteps with shape [B]
        :return: [B]
        '''
        bins = torch.linspace(0, 1, self.t_base, dtype=torch.float32).unsqueeze(0)
        t_grid = bins*t.float().unsqueeze(1)  # [B, num_bins]
        alpha_vals = alpha_fn(t_grid)

        if torch.any(alpha_vals<=0):
            raise ValueError("alpha(t) must > 0 to avoid log(≤0).")

        log_term = torch.log(alpha_vals)  # [B, num_bins]
        dt = t_grid[:, 1]-t_grid[:, 0]  # [B]
        integral = torch.cumsum((log_term[:, -1]+log_term[:, 1:])/2*dt.unsqueeze(1), dim=1)  # [B]
        x_vals = torch.exp(integral)
        return x_vals

    @staticmethod
    def alpha_bar_linear(beta_s, beta_e, t, N=1000):
        A = beta_e-beta_s
        B = 1-beta_s
        B_At = B-A*t

        # eps for stable
        eps = 1e-12
        B = torch.clamp(B, min=eps)
        B_At = torch.clamp(B_At, min=eps)

        term = (B*torch.log(B)-B_At*torch.log(B_At)-A*t)
        return torch.exp(N*term/A)

    @staticmethod
    def alpha_bar_scaled_linear(beta_s, beta_e, t, N=1000):
        sqrt_bs = torch.sqrt(beta_s)
        sqrt_be = torch.sqrt(beta_e)
        a = sqrt_be-sqrt_bs
        b = sqrt_bs
        u0 = b
        u1 = a*t+b

        eps = 1e-12

        def safe_log1m(u2):
            return torch.log(torch.clamp(1-u2, min=eps))

        def safe_log_frac(u):
            return torch.log(torch.clamp(1+u, min=eps))-torch.log(torch.clamp(1-u, min=eps))

        term1 = u1*safe_log1m(u1**2)
        term2 = 0.5*safe_log_frac(u1)
        term3 = u0*safe_log1m(u0**2)
        term4 = 0.5*safe_log_frac(u0)

        return torch.exp(N*(term1+term2-term3-term4)/a)

    def make_alpha_bar_fn(self, beta_schedule, beta_start, beta_end, alpha_fn=None):
        if alpha_fn is not None:
            return lambda t, alpha_fn_=alpha_fn:self.continuous_product(alpha_fn_(t), t)
        elif beta_schedule == "linear":
            return lambda t:self.alpha_bar_linear(beta_start, beta_end, t)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            return lambda t:self.alpha_bar_scaled_linear(beta_start, beta_end, t)
        elif beta_schedule == "squaredcos_cap_v2":
            return lambda t:torch.cos((t+0.008)/1.008*math.pi/2)**2
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            alpha_fn = lambda t:1-torch.sigmoid(torch.lerp(torch.full_like(t, -6), torch.full_like(t, 6), t))*(beta_end-beta_start)+beta_start
            return lambda t, alpha_fn_=alpha_fn:self.continuous_product(alpha_fn_(t), t)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented.")

    def sigma(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor([t])
        alpha_cumprod = self.alpha_bar_fn(t)
        return torch.sqrt(1-alpha_cumprod)

    def alpha(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor([t])
        alpha_cumprod = self.alpha_bar_fn(t)
        return torch.sqrt(alpha_cumprod)

    def c_noise(self, t: Union[float, torch.Tensor]):
        return t*self.t_base

    @property
    def sigma_start(self):
        return self.sigma(0)

    @property
    def sigma_end(self):
        return self.sigma(1)

    @property
    def alpha_start(self):
        return self.alpha(0)

    @property
    def alpha_end(self):
        return self.alpha(1)

    def alpha_to_t(self, alpha, t_min=0.0, t_max=1.0, tol=1e-5, max_iter=100):
        """
        alpha: [B]
        :return: t [B]
        """
        return invert_func(self.alpha, alpha, t_min, t_max, tol, max_iter)

    def sigma_to_t(self, sigma, t_min=0.0, t_max=1.0, tol=1e-5, max_iter=100):
        """
        sigma: [B]
        :return: t [B]
        """
        return invert_func(self.sigma, sigma, t_min, t_max, tol, max_iter)

class TimeSigmaScheduler(SigmaScheduler):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps

    def sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        '''
        :param t: 0-1, rate of time step
        '''
        if isinstance(t, float):
            t = torch.tensor(t)
        return ((t*self.num_timesteps).round().long()).clip(min=0, max=self.num_timesteps-1)

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        '''
        :param t: 0-1, rate of time step
        '''
        if isinstance(t, float):
            t = torch.tensor(t)
        return ((t*self.num_timesteps).round().long()).clip(min=0, max=self.num_timesteps-1)
    
    def c_noise(self, t: Union[float, torch.Tensor]):
        return (t*self.num_timesteps).round()

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
