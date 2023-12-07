import inspect
from typing import Dict, Any, Union, List

import torch
from torch.cuda.amp import autocast

from .base import BasicAction, from_memory_context, MemoryMixin

try:
    from diffusers.utils import randn_tensor
except:
    # new version of diffusers
    from diffusers.utils.torch_utils import randn_tensor

from hcpdiff.utils import prepare_seed
from hcpdiff.utils.net_utils import get_dtype
import random

class InputFeederAction(BasicAction):
    @from_memory_context
    def __init__(self, ex_inputs: Dict[str, Any], unet=None):
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
    def __init__(self, seed: Union[int, List[int]], bs: int = 1):
        super().__init__()
        self.seed = seed
        self.bs = bs

    def forward(self, device, **states):
        bs = states['prompt_embeds'].shape[0]//2 if 'prompt_embeds' in states else self.bs
        if self.seed is None:
            seeds = [None]*bs
        elif isinstance(self.seed, int):
            seeds = list(range(self.seed, self.seed+bs))
        else:
            seeds = self.seed
        seeds = [s or random.randint(0, 1 << 30) for s in seeds]

        G = prepare_seed(seeds, device=device)
        return {**states, 'seeds':seeds, 'generator':G, 'device':device}

class PrepareDiffusionAction(BasicAction, MemoryMixin):
    def __init__(self, dtype='fp32', amp=True):
        self.dtype = dtype
        self.amp = amp

    def forward(self, memory, **states):
        dtype = get_dtype(self.dtype)
        memory.unet.to(dtype=dtype)
        memory.text_encoder.to(dtype=dtype)
        memory.vae.to(dtype=dtype)

        device = memory.unet.device
        vae_scale_factor = 2**(len(memory.vae.config.block_out_channels)-1)
        return {**states, 'dtype':self.dtype, 'amp':self.amp, 'device':device, 'vae_scale_factor':vae_scale_factor}

class MakeTimestepsAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, scheduler=None, N_steps: int = 30, strength: float = None):
        self.scheduler = scheduler
        self.N_steps = N_steps
        self.strength = strength

    def get_timesteps(self, timesteps, strength):
        # get the original timestep using init_timestep
        num_inference_steps = len(timesteps)
        init_timestep = min(int(num_inference_steps*strength), num_inference_steps)

        t_start = max(num_inference_steps-init_timestep, 0)
        timesteps = timesteps[t_start*self.scheduler.order:]

        return timesteps

    def forward(self, memory, device, **states):
        self.scheduler = self.scheduler or memory.scheduler

        self.scheduler.set_timesteps(self.N_steps, device=device)
        timesteps = self.scheduler.timesteps
        if self.strength:
            timesteps = self.get_timesteps(timesteps, self.strength)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
        return {**states, 'device':device, 'timesteps':timesteps, 'alphas_cumprod':alphas_cumprod}

class MakeLatentAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, scheduler=None, N_ch=4, height=512, width=512):
        self.scheduler = scheduler
        self.N_ch = N_ch
        self.height = height
        self.width = width

    def forward(self, memory, generator, device, dtype, bs=None, latents=None, vae_scale_factor=8, start_timestep=None, **states):
        if bs is None:
            if 'prompt' in states:
                bs = len(states['prompt'])
        scheduler = self.scheduler or memory.scheduler

        shape = (bs, self.N_ch, self.height//vae_scale_factor, self.width//vae_scale_factor)
        if isinstance(generator, list) and len(generator) != bs:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {bs}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=get_dtype(dtype))
        if latents is None:
            # scale the initial noise by the standard deviation required by the scheduler
            latents = noise*scheduler.init_noise_sigma
        else:
            # image to image
            latents = latents.to(device)
            latents = scheduler.add_noise(latents, noise, start_timestep)

        return {**states, 'latents':latents, 'device':device, 'dtype':dtype, 'generator':generator}

class NoisePredAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, unet=None, scheduler=None, guidance_scale: float = 7.0):
        self.guidance_scale = guidance_scale
        self.unet = unet
        self.scheduler = scheduler

    def forward(self, memory, t, latents, prompt_embeds, pooled_output=None, encoder_attention_mask=None, crop_info=None,
                cross_attention_kwargs=None, dtype='fp32', amp=None, **states):
        self.scheduler = self.scheduler or memory.scheduler
        self.unet = self.unet or memory.unet

        with autocast(enabled=amp is not None, dtype=get_dtype(amp)):
            latent_model_input = torch.cat([latents]*2) if self.guidance_scale>1 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if pooled_output is None:
                noise_pred = self.unet(latent_model_input, t, prompt_embeds, encoder_attention_mask=encoder_attention_mask,
                                       cross_attention_kwargs=cross_attention_kwargs, ).sample
            else:
                added_cond_kwargs = {"text_embeds":pooled_output, "time_ids":crop_info}
                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, prompt_embeds, encoder_attention_mask=encoder_attention_mask,
                                       cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs).sample

            # perform guidance
            if self.guidance_scale>1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond+self.guidance_scale*(noise_pred_text-noise_pred_uncond)

        return {**states, 'noise_pred':noise_pred, 'latents':latents, 't':t, 'prompt_embeds':prompt_embeds, 'pooled_output':pooled_output,
            'crop_info':crop_info, 'cross_attention_kwargs':cross_attention_kwargs, 'dtype':dtype, 'amp':amp}

class SampleAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, scheduler=None, eta=0.0):
        self.scheduler = scheduler
        self.eta = eta

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def forward(self, memory, noise_pred, t, latents, generator, **states):
        self.scheduler = self.scheduler or memory.scheduler

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, self.eta)

        # compute the previous noisy sample x_t -> x_t-1
        sc_out = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
        latents = sc_out.prev_sample
        return {**states, 'latents':latents, 't':t, 'generator':generator}

class DiffusionStepAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, unet=None, scheduler=None, guidance_scale: float = 7.0):
        self.act_noise_pred = NoisePredAction(unet, scheduler, guidance_scale)
        self.act_sample = SampleAction(scheduler)

    def forward(self, memory, **states):
        states = self.act_noise_pred(memory=memory, **states)
        states = self.act_sample(memory=memory, **states)
        return states

class X0PredAction(BasicAction):
    def forward(self, latents, alphas_cumprod, t, noise_pred, **states):
        # x_t -> x_0
        alpha_prod_t = alphas_cumprod[t.long()]
        beta_prod_t = 1-alpha_prod_t
        latents_x0 = (latents-beta_prod_t**(0.5)*noise_pred)/alpha_prod_t**(0.5)  # approximate x_0
        return {**states, 'latents_x0':latents_x0, 'latents':latents, 'alphas_cumprod':alphas_cumprod, 't':t, 'noise_pred':noise_pred}
