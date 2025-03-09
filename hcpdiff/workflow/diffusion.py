import inspect
import random
import warnings
from typing import Dict, Any, Union, List

import torch
from hcpdiff.utils import prepare_seed
from hcpdiff.utils.net_utils import get_dtype, to_cpu, to_cuda
from rainbowneko.infer import BasicAction
from torch.cuda.amp import autocast

try:
    from diffusers.utils import randn_tensor
except:
    # new version of diffusers
    from diffusers.utils.torch_utils import randn_tensor

class InputFeederAction(BasicAction):
    def __init__(self, ex_inputs: Dict[str, Any], key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.ex_inputs = ex_inputs

    def forward(self, model, ex_inputs=None, **states):
        ex_inputs = self.ex_inputs if ex_inputs is None else {**ex_inputs, **self.ex_inputs}
        if hasattr(model, 'input_feeder'):
            for feeder in model.input_feeder:
                feeder(ex_inputs)

class SeedAction(BasicAction):
    def __init__(self, seed: Union[int, List[int]], bs: int = 1, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.seed = seed
        self.bs = bs

    def forward(self, device, gen_step=0, **states):
        bs = states['prompt_embeds'].shape[0]//2 if 'prompt_embeds' in states else self.bs
        if self.seed is None:
            seeds = [None]*bs
        elif isinstance(self.seed, int):
            seeds = list(range(self.seed+gen_step*bs, self.seed+(gen_step+1)*bs))
        else:
            seeds = self.seed
        seeds = [s or random.randint(0, 1 << 30) for s in seeds]

        G = prepare_seed(seeds, device=device)
        return {'seeds':seeds, 'generator':G}

class PrepareDiffusionAction(BasicAction):
    def __init__(self, model_offload=False, amp=torch.float16, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.model_offload = model_offload
        self.amp = amp

    def forward(self, device, unet, text_encoder, vae, **states):
        unet.to(device)
        text_encoder.to(device)
        vae.to(device)

        text_encoder.eval()
        unet.eval()
        vae.eval()
        return {'amp':self.amp, 'model_offload': self.model_offload}

class MakeTimestepsAction(BasicAction):
    def __init__(self, N_steps: int = 30, strength: float = None, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.N_steps = N_steps
        self.strength = strength

    def get_timesteps(self, scheduler, timesteps, strength):
        # get the original timestep using init_timestep
        num_inference_steps = len(timesteps)
        init_timestep = min(int(num_inference_steps*strength), num_inference_steps)

        t_start = max(num_inference_steps-init_timestep, 0)
        timesteps = timesteps[t_start*scheduler.order:]

        return timesteps

    def forward(self, scheduler, device, **states):

        scheduler.set_timesteps(self.N_steps, device=device)
        timesteps = scheduler.timesteps
        alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
        if self.strength:
            timesteps = self.get_timesteps(scheduler, timesteps, self.strength)
            return {'timesteps':timesteps, 'alphas_cumprod':alphas_cumprod, 'start_timestep':timesteps[:1]}
        else:
            return {'timesteps':timesteps, 'alphas_cumprod':alphas_cumprod}

class MakeLatentAction(BasicAction):
    def __init__(self, N_ch=4, height=None, width=None, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.N_ch = N_ch
        self.height = height
        self.width = width

    def forward(self, scheduler, vae, generator, device, dtype, bs=None, latents=None, start_timestep=None,
                pooled_output=None, crop_coord=None, **states):
        if bs is None:
            if 'prompt' in states:
                bs = len(states['prompt'])
        vae_scale_factor = 2**(len(vae.config.block_out_channels)-1)

        if latents is None:
            shape = (bs, self.N_ch, self.height//vae_scale_factor, self.width//vae_scale_factor)
        else:
            if self.height is not None:
                warnings.warn('latents exist! User-specified width and height will be ignored!')
            shape = latents.shape
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

        output = {'latents':latents}

        # SDXL inputs
        if pooled_output is not None:
            width, height = shape[3]*vae_scale_factor, shape[2]*vae_scale_factor
            if crop_coord is None:
                crop_info = torch.tensor([height, width, 0, 0, height, width], dtype=torch.float)
            else:
                crop_info = torch.tensor([height, width, *crop_coord], dtype=torch.float)
            crop_info = crop_info.to(device).repeat(bs, 1)
            output['text_embeds'] = pooled_output[-1].to(device)

            if 'negative_prompt' in states:
                output['crop_info'] = torch.cat([crop_info, crop_info], dim=0)

        return output


class DenoiseAction(BasicAction):
    def __init__(self, guidance_scale: float = 7.0, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.guidance_scale = guidance_scale

    def forward(self, unet, scheduler, t, latents, prompt_embeds, text_embeds=None, encoder_attention_mask=None, crop_info=None,
                cross_attention_kwargs=None, dtype='fp32', amp=None, model_offload=False, **states):

        if model_offload:
            to_cuda(unet) # to_cpu in VAE

        with autocast(enabled=amp is not None, dtype=get_dtype(amp)):
            latent_model_input = torch.cat([latents]*2) if self.guidance_scale>1 else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            if text_embeds is None:
                noise_pred = unet(latent_model_input, t, prompt_embeds, encoder_attention_mask=encoder_attention_mask,
                                       cross_attention_kwargs=cross_attention_kwargs, ).sample
            else:
                added_cond_kwargs = {"text_embeds":text_embeds, "time_ids":crop_info}
                # predict the noise residual
                noise_pred = unet(latent_model_input, t, prompt_embeds, encoder_attention_mask=encoder_attention_mask,
                                       cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs).sample

            # perform guidance
            if self.guidance_scale>1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond+self.guidance_scale*(noise_pred_text-noise_pred_uncond)

        return {'noise_pred':noise_pred}

class SampleAction(BasicAction):
    def __init__(self, eta=0.0, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.eta = eta

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

    def forward(self, scheduler, noise_pred, t, latents, generator, **states):

        extra_step_kwargs = self.prepare_extra_step_kwargs(scheduler, generator, self.eta)

        # compute the previous noisy sample x_t -> x_t-1
        sc_out = scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
        latents = sc_out.prev_sample
        return {'latents':latents}

class DiffusionStepAction(BasicAction):
    def __init__(self, guidance_scale: float = 7.0, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.act_noise_pred = DenoiseAction(guidance_scale)
        self.act_sample = SampleAction()

    def forward(self, unet, scheduler, **states):
        states = self.act_noise_pred(unet=unet, scheduler=scheduler, **states)
        states = self.act_sample(scheduler=scheduler, **states)
        return states

class X0PredAction(BasicAction):
    def forward(self, latents, alphas_cumprod, t, noise_pred, **states):
        # x_t -> x_0
        alpha_prod_t = alphas_cumprod[t.long()]
        beta_prod_t = 1-alpha_prod_t
        latents_x0 = (latents-beta_prod_t**(0.5)*noise_pred)/alpha_prod_t**(0.5)  # approximate x_0
        return {'latents_x0':latents_x0}

def time_iter(timesteps, **states):
    for t in timesteps:
        yield {'t':t}