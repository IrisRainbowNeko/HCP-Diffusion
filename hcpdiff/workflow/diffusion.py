import random
import warnings
from typing import Dict, Any, Union, List

import torch
from hcpdiff.diffusion.sampler import BaseSampler, DiffusersSampler
from hcpdiff.utils import prepare_seed
from hcpdiff.utils.net_utils import get_dtype, to_cuda
from rainbowneko.infer import BasicAction, Actions
from torch.cuda.amp import autocast
from einops import rearrange, repeat
from hcpdiff.models.compose import SDXLTextEncoder
from diffusers import FluxTransformer2DModel, PixArtTransformer2DModel

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

    def forward(self, device, seed=None, bs=None, **states):
        if bs is None:
            bs = states['prompt_embeds'].shape[0]//2 if 'prompt_embeds' in states else self.bs
        seed = seed or self.seed
        if seed is None:
            seeds = [None]*bs
        elif isinstance(seed, int):
            seeds = list(range(seed, seed+bs))
        else:
            seeds = seed
        seeds = [s or random.randint(0, 1 << 30) for s in seeds]

        G = prepare_seed(seeds, device=device)
        return {'seeds':seeds, 'generator':G}

class PrepareDiffusionAction(BasicAction):
    def __init__(self, model_offload=False, amp=torch.float16, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.model_offload = model_offload
        self.amp = amp

    def forward(self, device, denoiser, TE, vae, **states):
        denoiser.to(device)
        TE.to(device)
        vae.to(device)

        TE.eval()
        denoiser.eval()
        vae.eval()
        return {'amp':self.amp, 'model_offload':self.model_offload}

class MakeTimestepsAction(BasicAction):
    def __init__(self, N_steps: int = 30, strength: float = None, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.N_steps = N_steps
        self.strength = strength

    def get_timesteps(self, noise_sampler:BaseSampler, timesteps, strength):
        # get the original timestep using init_timestep
        num_inference_steps = len(timesteps)
        init_timestep = min(int(num_inference_steps*strength), num_inference_steps)

        t_start = max(num_inference_steps-init_timestep, 0)
        if isinstance(noise_sampler, DiffusersSampler):
            timesteps = timesteps[t_start*noise_sampler.scheduler.order:]
        else:
            timesteps = timesteps[t_start:]

        return timesteps

    def forward(self, noise_sampler:BaseSampler, device, **states):
        timesteps = noise_sampler.get_timesteps(self.N_steps, device=device)
        if self.strength:
            timesteps = self.get_timesteps(noise_sampler, timesteps, self.strength)
            return {'timesteps':timesteps, 'start_timestep':timesteps[:1]}
        else:
            return {'timesteps':timesteps}

class MakeLatentAction(BasicAction):
    def __init__(self, N_ch=4, height=None, width=None, patch_size=1, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.N_ch = N_ch
        self.height = height
        self.width = width
        self.patch_size = patch_size

    def forward(self, noise_sampler:BaseSampler, vae, generator, device, dtype, bs=None, latents=None, start_timestep=None,
                pooler_output=None, crop_coord=None, **states):
        if bs is None:
            if 'prompt' in states:
                bs = len(states['prompt'])
        vae_scale_factor = 2**(len(vae.config.block_out_channels)-1)
        device = torch.device(device)

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

        if latents is None:
            # scale the initial noise by the standard deviation required by the noise_sampler
            noise_sampler.generator = generator
            latents = noise_sampler.init_noise(shape, device=device, dtype=get_dtype(dtype))
            if self.patch_size>1:
                latents = rearrange(latents, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=self.patch_size, pw=self.patch_size)
        else:
            # image to image
            latents = latents.to(device)
            if self.patch_size>1:
                latents = rearrange(latents, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=self.patch_size, pw=self.patch_size)
            latents, noise = noise_sampler.add_noise(latents, start_timestep)

        output = {'latents':latents, 'latent_w':shape[3], 'latent_h':shape[2], 'patch_size':self.patch_size}

        # SDXL inputs
        if pooler_output is not None:
            width, height = shape[3]*vae_scale_factor, shape[2]*vae_scale_factor
            if crop_coord is None:
                crop_info = torch.tensor([height, width, 0, 0, height, width], dtype=torch.float)
            else:
                crop_info = torch.tensor([height, width, *crop_coord], dtype=torch.float)
            crop_info = crop_info.to(device).repeat(bs, 1)
            output['pooler_output'] = pooler_output.to(device)

            if 'negative_prompt' in states:
                output['crop_info'] = torch.cat([crop_info, crop_info], dim=0)

        return output

class SD15DenoiseAction(BasicAction):
    def __init__(self, guidance_scale: float = 7.0, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.guidance_scale = guidance_scale

    def forward(self, denoiser, noise_sampler: BaseSampler, t, latents, prompt_embeds, encoder_attention_mask=None,
                cross_attention_kwargs=None, dtype='fp32', amp=None, model_offload=False, **states):

        if model_offload:
            to_cuda(denoiser)  # to_cpu in VAE

        with autocast(enabled=amp is not None, dtype=get_dtype(amp)):
            latent_model_input = torch.cat([latents]*2) if self.guidance_scale>1 else latents
            latent_model_input = noise_sampler.sigma_scheduler.c_in(t)*latent_model_input
            t_in = noise_sampler.sigma_scheduler.c_noise(t)

            noise_pred = denoiser(latent_model_input, t_in, prompt_embeds, encoder_attention_mask=encoder_attention_mask,
                                cross_attention_kwargs=cross_attention_kwargs, ).sample
            # perform guidance
            if self.guidance_scale>1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond+self.guidance_scale*(noise_pred_text-noise_pred_uncond)

        return {'noise_pred':noise_pred}

class SDXLDenoiseAction(BasicAction):
    def __init__(self, guidance_scale: float = 7.0, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.guidance_scale = guidance_scale

    def forward(self, denoiser, noise_sampler: BaseSampler, t, latents, prompt_embeds, pooler_output=None, encoder_attention_mask=None,
                crop_info=None, cross_attention_kwargs=None, dtype='fp32', amp=None, model_offload=False, **states):

        if model_offload:
            to_cuda(denoiser)  # to_cpu in VAE

        with autocast(enabled=amp is not None, dtype=get_dtype(amp)):
            latent_model_input = torch.cat([latents]*2) if self.guidance_scale>1 else latents
            latent_model_input = noise_sampler.sigma_scheduler.c_in(t)*latent_model_input
            t_in = noise_sampler.sigma_scheduler.c_noise(t)

            added_cond_kwargs = {"text_embeds":pooler_output, "time_ids":crop_info}
            # predict the noise residual
            noise_pred = denoiser(latent_model_input, t_in, prompt_embeds, encoder_attention_mask=encoder_attention_mask,
                                cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs).sample

            # perform guidance
            if self.guidance_scale>1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond+self.guidance_scale*(noise_pred_text-noise_pred_uncond)

        return {'noise_pred':noise_pred}

class PixartDenoiseAction(BasicAction):
    def __init__(self, guidance_scale: float = 7.0, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.guidance_scale = guidance_scale

    def forward(self, denoiser, noise_sampler: BaseSampler, t, latents, prompt_embeds, encoder_attention_mask=None,
                cross_attention_kwargs=None, dtype='fp32', amp=None, model_offload=False, **states):

        if model_offload:
            to_cuda(denoiser)  # to_cpu in VAE

        with autocast(enabled=amp is not None, dtype=get_dtype(amp)):
            latent_model_input = torch.cat([latents]*2) if self.guidance_scale>1 else latents
            latent_model_input = noise_sampler.sigma_scheduler.c_in(t)*latent_model_input
            t_in = noise_sampler.sigma_scheduler.c_noise(t)

            if t_in.dim() == 0:
                t_in = t_in.unsqueeze(0).expand(latent_model_input.shape[0])
            
            noise_pred = denoiser(latent_model_input, prompt_embeds, t_in, encoder_attention_mask=encoder_attention_mask,
                                cross_attention_kwargs=cross_attention_kwargs, ).sample
            # perform guidance
            if self.guidance_scale>1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond+self.guidance_scale*(noise_pred_text-noise_pred_uncond)
        
        # remove vars from DiT
        noise_pred, _ = noise_pred.chunk(2, dim=1)

        return {'noise_pred':noise_pred}

class FluxDenoiseAction(BasicAction):
    def __init__(self, guidance_scale: float = 7.0, true_cfg=False, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.guidance_scale = guidance_scale
        self.true_cfg = true_cfg

    def forward(self, denoiser, noise_sampler: BaseSampler, t, latents, prompt_embeds, pooler_output=None, encoder_attention_mask=None,
                latent_w=None, latent_h=None, dtype='fp32', amp=None, model_offload=False, **states):

        if model_offload:
            to_cuda(denoiser)  # to_cpu in VAE

        with autocast(enabled=amp is not None, dtype=get_dtype(amp)):
            if self.true_cfg:
                latent_model_input = torch.cat([latents]*2) if self.guidance_scale>1 else latents
                latent_model_input = noise_sampler.sigma_scheduler.c_in(t)*latent_model_input
                t_in = noise_sampler.sigma_scheduler.c_noise(t)
                latent_model_input = rearrange(latent_model_input, "b c h w -> b (h w) c")

                img_ids = torch.zeros(latent_h, latent_w, 3)
                img_ids[..., 1] = img_ids[..., 1]+torch.arange(latent_h)[:, None]
                img_ids[..., 2] = img_ids[..., 2]+torch.arange(latent_w)[None, :]
                img_ids = repeat(img_ids, "h w c -> b (h w) c", b=latent_model_input.shape[0])

                txt_ids = torch.zeros(prompt_embeds.shape[0], prompt_embeds.shape[1], 3)

                # predict the noise residual
                noise_pred = denoiser(latent_model_input, t_in, 1.0, pooler_output, prompt_embeds, txt_ids, img_ids).sample

                # perform guidance
                if self.guidance_scale>1:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond+self.guidance_scale*(noise_pred_text-noise_pred_uncond)
            else:
                latent_model_input = latents
                latent_model_input = noise_sampler.sigma_scheduler.c_in(t)*latent_model_input
                t_in = noise_sampler.sigma_scheduler.c_noise(t)
                latent_model_input = rearrange(latent_model_input, "b c h w -> b (h w) c")

                img_ids = torch.zeros(latent_h, latent_w, 3)
                img_ids[..., 1] = img_ids[..., 1]+torch.arange(latent_h)[:, None]
                img_ids[..., 2] = img_ids[..., 2]+torch.arange(latent_w)[None, :]
                img_ids = repeat(img_ids, "h w c -> b (h w) c", b=latent_model_input.shape[0])

                txt_ids = torch.zeros(latent_model_input.shape[0], prompt_embeds.shape[1], 3)

                # predict the noise residual
                noise_pred = denoiser(latent_model_input, t_in, self.guidance_scale, pooler_output, prompt_embeds, txt_ids, img_ids).sample
            noise_pred = rearrange(noise_pred, "b (h w) c -> b c h w", h=latent_h, w=latent_w)

        return {'noise_pred':noise_pred}

class SampleAction(BasicAction):
    def forward(self, noise_sampler: BaseSampler, noise_pred, t, latents, generator, **states):
        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_sampler.denoise(latents, t, noise_pred, generator=generator)
        return {'latents':latents}

class DiffusionStepAction(BasicAction):
    def __init__(self, guidance_scale: float = 7.0, denoise_action:str|BasicAction='auto', true_cfg=False, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        if callable(denoise_action):
            self.act_noise_pred = denoise_action(guidance_scale)
        else:
            self.act_noise_pred = None
            self.true_cfg = true_cfg
            self.guidance_scale = guidance_scale
        self.act_sample = SampleAction()

    def forward(self, denoiser, noise_sampler, TE, **states):
        if self.act_noise_pred is None:
            if isinstance(denoiser, FluxTransformer2DModel):
                self.act_noise_pred = FluxDenoiseAction(guidance_scale=self.guidance_scale, true_cfg=self.true_cfg)
            elif isinstance(TE, SDXLTextEncoder):
                self.act_noise_pred = SDXLDenoiseAction(guidance_scale=self.guidance_scale)
            elif isinstance(denoiser, PixArtTransformer2DModel):
                self.act_noise_pred = PixartDenoiseAction(guidance_scale=self.guidance_scale)
            else:
                self.act_noise_pred = SD15DenoiseAction(guidance_scale=self.guidance_scale)

        states = self.act_noise_pred(denoiser=denoiser, noise_sampler=noise_sampler, **states)
        states = self.act_sample(**states)
        return states
    
class DiffusionActions(Actions):
    def __init__(self, actions: List[BasicAction], clean_latent=True, seed_inc=True, key_map_in=None, key_map_out=None):
        super().__init__(actions, key_map_in=key_map_in, key_map_out=key_map_out)
        self.clean_latent = clean_latent
        self.seed_inc = seed_inc

    def forward(self, **states):
        states = super().forward(**states)
        if self.seed_inc and 'seed' in states:
            bs = states['latents'].shape[0]
            states['seed'] = states['seed'] + bs
        if self.clean_latent:
            states.pop('noise_pred', None)
            states.pop('latents', None)
            states.pop('prompt', None)
            states.pop('negative_prompt', None)
        return states

class X0PredAction(BasicAction):
    def forward(self, latents, noise_sampler: BaseSampler, t, noise_pred, **states):
        latents_x0 = noise_sampler.pred_for_target(noise_pred, latents, t, target_type='x0')
        return {'latents_x0':latents_x0}

def time_iter(timesteps, **states):
    return [{'t':t} for t in timesteps]
