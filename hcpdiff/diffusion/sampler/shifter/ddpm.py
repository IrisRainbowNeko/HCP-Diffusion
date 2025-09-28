import math
from typing import Callable

import torch

class DDPMDiscreteShifter:
    def __init__(self, beta_schedule: str = "scaled_linear", linear_start=0.00085, linear_end=0.0120, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        self.betas = self.make_betas(beta_schedule, linear_start, linear_end, num_timesteps)
        alphas = 1.0-self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def __call__(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.alphas_cumprod[((t*self.num_timesteps).round().long()).clip(min=0, max=self.num_timesteps-1)]

    @property
    def min_dt(self):
        return 1/self.num_timesteps

    def betas_for_alpha_bar(
        self,
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

    def make_betas(self, beta_schedule, beta_start, beta_end, num_train_timesteps, betas=None):
        if betas is not None:
            return torch.tensor(betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            return torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32)**2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            return self.betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            return torch.sigmoid(betas)*(beta_end-beta_start)+beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented.")

class DDPMContinuousShifter:
    def __init__(self, beta_schedule: str = "scaled_linear", linear_start=0.00085, linear_end=0.0120, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        self.alphas_cumprod_fn = self.make_alphas_cumprod_fn(beta_schedule, linear_start, linear_end)

    def __call__(self, t: torch.Tensor, **kwargs):
        return self.alphas_cumprod_fn(t)

    @property
    def min_dt(self):
        return 1e-8

    def continuous_product(self, alpha_fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor):
        '''

        :param alpha_fn: alpha function
        :param t: timesteps with shape [B]
        :return: [B]
        '''
        bins = torch.linspace(0, 1, self.num_timesteps, dtype=torch.float32).unsqueeze(0)
        t_grid = bins*t.float().unsqueeze(1)  # [B, num_bins]
        alphas = alpha_fn(t_grid)

        if torch.any(alphas<=0):
            raise ValueError("alpha(t) must > 0 to avoid log(≤0).")

        log_alphas = torch.log(alphas)  # [B, num_bins]
        dt = t_grid[:, 1]-t_grid[:, 0]  # [B]
        integral = torch.cumsum((log_alphas[:, -1]+log_alphas[:, 1:])/2*dt.unsqueeze(1), dim=1)  # [B]
        x_vals = torch.exp(integral)
        euler_maclaurin = (log_alphas[:,0]+log_alphas[:,-1])/2
        return x_vals + euler_maclaurin

    def alpha_bar_linear(self, beta_s, beta_e, t, eps = 1e-12):
        alpha = lambda t: 1-t*(beta_e-beta_s)-beta_s
        def F(t):
            alpha_t = alpha(t).clamp(min=eps)
            return alpha_t * alpha_t.log() - alpha_t

        riemann_term = -(self.num_timesteps/(beta_e-beta_s))*(F(t) - F(0))
        euler_maclaurin_term = (alpha(t).log()+alpha(0).log())/2
        return torch.exp(riemann_term + euler_maclaurin_term)

    def alpha_bar_scaled_linear(self, beta_s, beta_e, t, eps = 1e-12):
        beta_s_sqrt = beta_s.sqrt()
        beta_e_sqrt = beta_e.sqrt()
        v = lambda t: t*(beta_e_sqrt-beta_s_sqrt)+beta_s_sqrt
        alpha = lambda t: 1-v(t)**2
        k = beta_e_sqrt - beta_s_sqrt
        def F(t):
            v_t = v(t).clamp(min=eps)
            return v_t * (1-v_t**2).log() + 2*torch.arctanh(v_t) - 2*v_t

        riemann_term = self.num_timesteps/k * (F(t) - F(0))
        euler_maclaurin_term = (alpha(t).log()+alpha(0).log())/2
        return torch.exp(riemann_term + euler_maclaurin_term)

    def make_alphas_cumprod_fn(self, beta_schedule, beta_start, beta_end, alpha_fn=None):
        if isinstance(beta_start, float):
            beta_start = torch.tensor(beta_start, dtype=torch.float32)
        if isinstance(beta_end, float):
            beta_end = torch.tensor(beta_end, dtype=torch.float32)

        if alpha_fn is not None:
            return lambda t, alpha_fn_=alpha_fn:self.continuous_product(alpha_fn_(t), t)
        elif beta_schedule == "linear":
            return lambda t:self.alpha_bar_linear(beta_start, beta_end, t)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            return lambda t:self.alpha_bar_scaled_linear(beta_start, beta_end, t)
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            return lambda t:torch.cos((t+0.008)/1.008*math.pi/2)**2
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            alpha_fn = lambda t:1-torch.sigmoid(torch.lerp(torch.full_like(t, -6), torch.full_like(t, 6), t))*(beta_end-beta_start)+beta_start
            return lambda t, alpha_fn_=alpha_fn:self.continuous_product(alpha_fn_(t), t)
        else:
            raise NotImplementedError(f"{beta_schedule} does not implemented.")