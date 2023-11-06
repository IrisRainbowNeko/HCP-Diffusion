from .base import BasicAction, from_memory_context, ExecAction, MemoryMixin
from typing import Dict, Any, Union, List
import torch
from diffusers.utils import randn_tensor
from hcpdiff.utils import prepare_seed
import random

class InputFeederAction(BasicAction):
    @from_memory_context
    def __init__(self, ex_inputs:Dict[str, Any], unet=None):
        super().__init__()
        self.ex_inputs = ex_inputs
        self.unet = unet

    def forward(self, **states):
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(self.ex_inputs)
        return states

class SeedAction(BasicAction):
    @from_memory_context
    def __init__(self, seed:Union[int, List[int]], bs:int=1):
        super().__init__()
        self.seed = seed
        self.bs = bs

    def forward(self, **states):
        bs = states['prompt_embeds'].shape[0]//2 if 'prompt_embeds' in states else self.bs
        if self.seed is None:
            seeds = [None]*bs
        elif isinstance(self.seed, int):
            seeds = list(range(self.seed, self.seed+bs))
        else:
            seeds = self.seed
        seeds = [s or random.randint(0, 1 << 30) for s in seeds]

        G = prepare_seed(seeds)
        return {'seeds':seeds, 'generator':G, **states}

class PrepareDiffusionAction(BasicAction, MemoryMixin):
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def forward(self, memory, **states):
        device = memory.unet.device
        vae_scale_factor = 2**(len(memory.vae.config.block_out_channels)-1)
        return {'dtype': self.dtype, 'device':device, 'vae_scale_factor':vae_scale_factor, **states}

class MakeTimestepsAction(BasicAction):
    @from_memory_context
    def __init__(self, scheduler, N_steps:int=30, strength:float=None):
        self.scheduler = scheduler
        self.N_steps = N_steps
        self.strength = strength

    def get_timesteps(self, timesteps, strength):
        # get the original timestep using init_timestep
        num_inference_steps = len(timesteps)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order :]

        return timesteps

    def forward(self, device, **states):
        self.scheduler.set_timesteps(self.N_steps, device=device)
        timesteps = self.scheduler.timesteps
        if self.strength:
            timesteps = self.get_timesteps(timesteps, self.strength)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
        return {'device':device, 'timesteps':timesteps, 'alphas_cumprod':alphas_cumprod, **states}

class MakeLatentAction(BasicAction):
    @from_memory_context
    def __init__(self, scheduler=None, N_ch=4, height=512, width=512):
        self.scheduler = scheduler
        self.N_ch=N_ch
        self.height=height
        self.width=width

    def forward(self, generator, device, dtype, bs=None, latents=None, vae_scale_factor=8, start_timestep=None, **states):
        if bs is None:
            if 'prompt' in states:
                bs = len(states['prompt'])
        scheduler = self.scheduler or states['scheduler']

        shape = (bs, self.N_ch, self.height//vae_scale_factor, self.width//vae_scale_factor)
        if isinstance(generator, list) and len(generator) != bs:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {bs}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if latents is None:
            # scale the initial noise by the standard deviation required by the scheduler
            latents = noise*scheduler.init_noise_sigma
        else:
            # image to image
            latents = latents.to(device)
            latents = scheduler.add_noise(latents, noise, start_timestep)

        return {'latents': latents, 'device':device, 'dtype':dtype, 'generator':generator, **states}

class NoisePredAction(BasicAction):
    @from_memory_context
    def __init__(self, unet, scheduler, guidance_scale:float=7.0):
        self.guidance_scale=guidance_scale
        self.unet = unet
        self.scheduler = scheduler

    def forward(self, t, latents, prompt_embeds, device, dtype, pooled_output=None, crop_info=None, cross_attention_kwargs=None, **states):
        latent_model_input = torch.cat([latents]*2) if self.guidance_scale>1 else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        if pooled_output is None:
            noise_pred = self.unet(latent_model_input, t, prompt_embeds,
                                   cross_attention_kwargs=cross_attention_kwargs, ).sample
        else:
            added_cond_kwargs = {"text_embeds":pooled_output, "time_ids":crop_info}
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, prompt_embeds,
                                   cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs).sample

        # perform guidance
        if self.guidance_scale>1:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond+self.guidance_scale*(noise_pred_text-noise_pred_uncond)

        return {'noise_pred':noise_pred, 'latents': latents, 't':t, 'prompt_embeds':prompt_embeds, 'device':device, 'dtype':dtype,
            'pooled_output':pooled_output, 'crop_info':crop_info, 'cross_attention_kwargs':cross_attention_kwargs, **states}

class SampleAction(BasicAction):
    @from_memory_context
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def forward(self, noise_pred, t, latents, extra_step_kwargs={}, **states):
        # compute the previous noisy sample x_t -> x_t-1
        sc_out = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
        latents = sc_out.prev_sample
        return {'latents': latents, 't':t, 'extra_step_kwargs':extra_step_kwargs, **states}

class DiffusionStepAction(BasicAction):
    @from_memory_context
    def __init__(self, unet, scheduler, guidance_scale:float=7.0):
        self.act_noise_pred = NoisePredAction(unet, scheduler, guidance_scale)
        self.act_sample = SampleAction(scheduler)

    def forward(self, **states):
        states = self.act_noise_pred(**states)
        states = self.act_sample(**states)
        return states

class X0PredAction(BasicAction):
    def forward(self, latents, alphas_cumprod, t, noise_pred, **states):
        # x_t -> x_0
        alpha_prod_t = alphas_cumprod[t.long()]
        beta_prod_t = 1-alpha_prod_t
        latents_x0 = (latents-beta_prod_t**(0.5)*noise_pred)/alpha_prod_t**(0.5)  # approximate x_0
        return {'latents_x0': latents_x0, 'latents': latents, 'alphas_cumprod':alphas_cumprod, 't':t, 'noise_pred':noise_pred, **states}