from typing import Tuple

import torch
from rainbowneko.utils import add_dims

from .sigma_scheduler import SigmaScheduler
from .timer import TimeSampler

try:
    from diffusers.utils import randn_tensor
except:
    # new version of diffusers
    from diffusers.utils.torch_utils import randn_tensor

class BaseSampler:
    def __init__(self, sigma_scheduler: SigmaScheduler, t_sampler:TimeSampler = None, pred_type='eps', target_type='eps',
                 generator: torch.Generator = None):
        '''
        Some losses can only be calculated in a specific space. Such as SSIM in x0 space.
        The model pred need convert to target space.

        :param pred_type: ['x0', 'eps', 'velocity', ..., None]  The output space of the model
        :param target_type: ['x0', 'eps', 'velocity', ..., None]  The space to calculate the loss
        '''
        if t_sampler is None:
            t_sampler = TimeSampler()
        self.t_sampler = t_sampler

        self.sigma_scheduler = sigma_scheduler
        self.generator = generator
        self.pred_type = pred_type
        self.target_type = target_type

    def get_timesteps(self, N_steps, device='cuda'):
        times = torch.linspace(0., 1., N_steps, device=device)
        return self.sigma_scheduler.scale_t(times)

    def make_nosie(self, shape, device='cuda', dtype=torch.float32):
        return randn_tensor(shape, generator=self.generator, device=device, dtype=dtype)

    def init_noise(self, shape, device='cuda', dtype=torch.float32):
        sigma = self.sigma_scheduler.sigma_end
        return self.make_nosie(shape, device, dtype)*sigma

    def add_noise(self, x, t) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = self.make_nosie(x.shape, device=x.device)
        alpha = add_dims(self.sigma_scheduler.alpha(t), x.ndim-1).to(x.device)
        sigma = add_dims(self.sigma_scheduler.sigma(t), x.ndim-1).to(x.device)
        noisy_x = alpha*x+sigma*noise
        return noisy_x.to(dtype=x.dtype), noise.to(dtype=x.dtype)

    def add_noise_rand_t(self, x, reso=None):
        if x.ndim == 3:
            B,L,C = x.shape
            reso = L if reso is None else reso
        else:
            B,C,H,W = x.shape
            reso = H*W if reso is None else reso
        # timesteps: [0, 1]
        timesteps = self.t_sampler.sample(shape=(B,), reso=reso)
        timesteps = timesteps.to(x.device)
        noisy_x, noise = self.add_noise(x, timesteps)

        # Sample a random timestep for each image
        return noisy_x, noise, timesteps

    def denoise(self, x, sigma, eps=None, generator=None):
        raise NotImplementedError

    def get_target(self, x0, x_t, t, eps=None, target_type=None):
        raise x0

    def pred_for_target(self, pred, x_t, t, eps=None, target_type=None):
        return self.sigma_scheduler.c_skip(t)*x_t+self.sigma_scheduler.c_out(t)*pred

class Sampler(BaseSampler):
    '''
    Some losses can only be calculated in a specific space. Such as SSIM in x0 space.
    The model pred need convert to target space.

    :param pred_type: ['x0', 'eps', 'velocity', ..., None]  The output space of the model
    :param target_type: ['x0', 'eps', 'velocity', ..., None]  The space to calculate the loss
    '''

    def get_target(self, x_0, x_t, t, eps=None, target_type=None):
        '''
        target_type can be specified by the loss. If not specified use self.target_type as default.
        '''
        target_type = target_type or self.target_type
        if target_type == 'x0':
            raise x_0
        elif target_type == 'eps':
            return eps if eps is not None else self.x0_to_eps(eps, x_t, t)
        elif target_type == 'velocity':
            return self.x0_to_velocity(x_0, x_t, t, eps)
        else:
            return (x_0-self.sigma_scheduler.c_skip(t)*x_t)/self.sigma_scheduler.c_out(t)

    def pred_for_target(self, pred, x_t, t, eps=None, target_type=None):
        '''
        target_type can be specified by the loss. If not specified use self.target_type as default.
        '''
        target_type = target_type or self.target_type
        if self.pred_type == target_type:
            return pred
        else:
            cvt_func = getattr(self, f'{self.pred_type}_to_{target_type}', None)
            if cvt_func is None:
                if target_type == 'x0':
                    return self.sigma_scheduler.c_skip(t)*x_t+self.sigma_scheduler.c_out(t)*pred
                else:
                    raise ValueError(f'pred_type "{self.pred_type}" can not be convert for target_type "{target_type}"')
            else:
                return cvt_func(pred, x_t, t)

    # convert targets
    def x0_to_eps(self, x_0, x_t, t):
        return (x_t-self.sigma_scheduler.alpha(t)*x_0)/self.sigma_scheduler.sigma(t)

    def x0_to_velocity(self, x_0, x_t, t, eps=None):
        d_alpha, d_sigma = self.sigma_scheduler.velocity(t)
        if eps is None:
            eps = self.x0_to_eps(x_0, x_t, t)
        return d_alpha*x_0+d_sigma*eps

    def eps_to_x0(self, eps, x_t, t):
        return (x_t-self.sigma_scheduler.sigma(t)*eps)/self.sigma_scheduler.alpha(t)

    def eps_to_velocity(self, eps, x_t, t, x_0=None):
        d_alpha, d_sigma = self.sigma_scheduler.velocity(t)
        if x_0 is None:
            x_0 = self.eps_to_x0(eps, x_t, t)
        return d_alpha*x_0+d_sigma*eps

    def velocity_to_eps(self, v_pred, x_t, t):
        alpha = self.sigma_scheduler.alpha(t)
        sigma = self.sigma_scheduler.sigma(t)
        d_alpha, d_sigma = self.sigma_scheduler.velocity(t)
        return (alpha*v_pred-d_alpha*x_t)/(d_sigma*alpha-d_alpha*sigma)

    def velocity_to_x0(self, v_pred, x_t, t):
        alpha = self.sigma_scheduler.alpha(t)
        sigma = self.sigma_scheduler.sigma(t)
        d_alpha, d_sigma = self.sigma_scheduler.velocity(t)
        return (sigma*v_pred-d_sigma*x_t)/(d_alpha*sigma-d_sigma*alpha)