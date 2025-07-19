from contextlib import nullcontext
from functools import partial
from typing import Dict, Union

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from rainbowneko.models.wrapper import BaseWrapper
from torch import Tensor
from torch import nn

from hcpdiff.diffusion.sampler import BaseSampler
from hcpdiff.models import TEEXHook
from hcpdiff.models.compose import ComposeTEEXHook
from hcpdiff.utils import pad_attn_bias
from .utils import TEHookCFG, SD15_TEHookCFG, SDXL_TEHookCFG
from ..cfg_context import CFGContext

class SD15Wrapper(BaseWrapper):
    def __init__(self, denoiser: UNet2DConditionModel, TE, vae: AutoencoderKL, noise_sampler: BaseSampler, tokenizer, min_attnmask=0,
                 TE_hook_cfg:TEHookCFG=SD15_TEHookCFG, cfg_context=CFGContext(), key_map_in=None, key_map_out=None):
        super().__init__()
        self.key_mapper_in = self.build_mapper(key_map_in, None, (
            'prompt -> prompt_ids', 'image -> image', 'attn_mask -> attn_mask', 'position_ids -> position_ids', 'neg_prompt -> neg_prompt_ids',
            'neg_attn_mask -> neg_attn_mask', 'neg_position_ids -> neg_position_ids', 'plugin_input -> plugin_input'))
        self.key_mapper_out = self.build_mapper(key_map_out, None, None)

        self.denoiser = denoiser
        self.TE = TE
        self.vae = vae
        self.noise_sampler = noise_sampler
        self.tokenizer = tokenizer
        self.min_attnmask = min_attnmask

        self.TE_hook_cfg = TEHookCFG.create(TE_hook_cfg)
        self.cfg_context = cfg_context
        self.tokenizer.N_repeats = self.TE_hook_cfg.tokenizer_repeats

    def post_init(self):
        self.make_TE_hook(self.TE_hook_cfg)

        self.vae_trainable = False
        if self.vae is not None:
            for p in self.vae.parameters():
                if p.requires_grad:
                    self.vae_trainable = True
                    break

        self.TE_trainable = False
        for p in self.TE.parameters():
            if p.requires_grad:
                self.TE_trainable = True
                break

    def make_TE_hook(self, TE_hook_cfg):
        # Hook and extend text_encoder
        self.text_enc_hook = TEEXHook.hook(self.TE, self.tokenizer, N_repeats=TE_hook_cfg.tokenizer_repeats,
                                           clip_skip=TE_hook_cfg.clip_skip, clip_final_norm=TE_hook_cfg.clip_final_norm)

    def get_latents(self, image: Tensor):
        if image.shape[1] == 3:
            with torch.no_grad() if self.vae_trainable else nullcontext():
                latents = self.vae.encode(image.to(dtype=self.vae.dtype)).latent_dist.sample()
                if shift_factor := getattr(self.vae.config, 'shift_factor', None) is not None:
                    latents = (latents-shift_factor)*self.vae.config.scaling_factor
                else:
                    latents = latents*self.vae.config.scaling_factor
        else:
            latents = image  # Cached latents
        return latents

    def forward_TE(self, prompt_ids, timesteps, attn_mask=None, position_ids=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask, **plugin_input)
        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        # Get the text embedding for conditioning
        encoder_hidden_states = self.TE(prompt_ids, position_ids=position_ids, attention_mask=attn_mask, output_hidden_states=True)[0]
        return encoder_hidden_states

    def forward_denoiser(self, x_t, prompt_ids, encoder_hidden_states, timesteps, attn_mask=None, position_ids=None, plugin_input={}, **kwargs):
        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask,
                         encoder_hidden_states=encoder_hidden_states, **plugin_input)
        if hasattr(self.denoiser, 'input_feeder'):
            for feeder in self.denoiser.input_feeder:
                feeder(input_all)
        model_pred = self.denoiser(x_t, timesteps, encoder_hidden_states, encoder_attention_mask=attn_mask).sample  # Predict the noise residual
        return model_pred

    def pn_cat(self, neg, pos, dim=0):
        if isinstance(pos, dict): # ComposeTextEncoder
            return {name:torch.cat([neg[name], pos_i], dim=dim) for name, pos_i in pos.items()}
        else:
            return torch.cat([neg, pos], dim=dim)

    def model_forward(self, prompt_ids, image, attn_mask=None, position_ids=None, neg_prompt_ids=None, neg_attn_mask=None, neg_position_ids=None,
                      plugin_input={}, **kwargs):
        # input prepare
        x_0 = self.get_latents(image)
        x_t, noise, timesteps = self.noise_sampler.add_noise_rand_t(x_0)
        x_t_in = x_t*self.noise_sampler.sigma_scheduler.c_in(timesteps).to(dtype=x_t.dtype).view(-1,1,1,1)
        t_in = self.noise_sampler.sigma_scheduler.c_noise(timesteps)

        if neg_prompt_ids:
            prompt_ids = self.pn_cat(neg_prompt_ids, prompt_ids)
            if neg_attn_mask:
                attn_mask = self.pn_cat(neg_attn_mask, attn_mask)
            if neg_position_ids:
                position_ids = self.pn_cat(neg_position_ids, position_ids)

        # model forward
        x_t_in, t_in = self.cfg_context.pre(x_t_in, t_in)
        encoder_hidden_states = self.forward_TE(prompt_ids, t_in, attn_mask=attn_mask, position_ids=position_ids,
                                                plugin_input=plugin_input, **kwargs)
        model_pred = self.forward_denoiser(x_t_in, prompt_ids, encoder_hidden_states, t_in, attn_mask=attn_mask, position_ids=position_ids,
                                           plugin_input=plugin_input, **kwargs)
        model_pred = self.cfg_context.post(model_pred)

        return dict(model_pred=model_pred, noise=noise, timesteps=timesteps, x_0=x_0, x_t=x_t, noise_sampler=self.noise_sampler)

    def forward(self, ds_name=None, **kwargs):
        model_args, model_kwargs = self.get_map_data(self.key_mapper_in, kwargs, ds_name)
        out = self.model_forward(*model_args, **model_kwargs)
        return self.get_map_data(self.key_mapper_out, out, ds_name=ds_name)[1]

    def enable_gradient_checkpointing(self):
        def grad_ckpt_enable(m):
            if getattr(m, 'gradient_checkpointing', False):
                m.training = True

        self.denoiser.enable_gradient_checkpointing()
        if self.TE_trainable:
            self.TE.gradient_checkpointing_enable()
        self.apply(grad_ckpt_enable)

    def enable_xformers(self):
        self.denoiser.enable_xformers_memory_efficient_attention()

    @property
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def trainable_models(self) -> Dict[str, nn.Module]:
        return {'self':self}

    def set_dtype(self, dtype, vae_dtype):
        self.dtype = dtype
        self.vae_dtype = vae_dtype
        # Move vae and text_encoder to device and cast to weight_dtype
        if self.vae is not None:
            self.vae = self.vae.to(dtype=vae_dtype)
        if not self.TE_trainable:
            self.TE = self.TE.to(dtype=dtype)

    @classmethod
    def from_pretrained(cls, models: Union[partial, Dict[str, nn.Module]], **kwargs):
        models = models() if isinstance(models, partial) else models
        return cls(models['denoiser'], models['TE'], models['vae'], models['noise_sampler'], models['tokenizer'], **kwargs)

class SDXLWrapper(SD15Wrapper):
    def __init__(self, denoiser: UNet2DConditionModel, TE, vae: AutoencoderKL, noise_sampler: BaseSampler, tokenizer, min_attnmask=0,
                 TE_hook_cfg:TEHookCFG=SDXL_TEHookCFG, cfg_context=CFGContext(), key_map_in=None, key_map_out=None):
        super().__init__(denoiser, TE, vae, noise_sampler, tokenizer, min_attnmask, TE_hook_cfg, cfg_context, key_map_in, key_map_out)
        self.key_mapper_in = self.build_mapper(key_map_in, None, (
            'prompt -> prompt_ids', 'image -> image', 'attn_mask -> attn_mask', 'position_ids -> position_ids', 'neg_prompt -> neg_prompt_ids',
            'neg_attn_mask -> neg_attn_mask', 'neg_position_ids -> neg_position_ids', 'plugin_input -> plugin_input', 'coord -> crop_info'))

    def make_TE_hook(self, TE_hook_cfg):
        # Hook and extend text_encoder
        self.text_enc_hook = ComposeTEEXHook.hook(self.TE, self.tokenizer, N_repeats=TE_hook_cfg.tokenizer_repeats,
                                                  clip_skip=TE_hook_cfg.clip_skip, clip_final_norm=TE_hook_cfg.clip_final_norm)

    def forward_TE(self, prompt_ids, timesteps, attn_mask=None, position_ids=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask, **plugin_input)
        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        # Get the text embedding for conditioning
        encoder_hidden_states, pooled_output = self.TE(prompt_ids, position_ids=position_ids, attention_mask=attn_mask, output_hidden_states=True)
        return encoder_hidden_states, pooled_output

    def forward_denoiser(self, x_t, prompt_ids, encoder_hidden_states, timesteps, added_cond_kwargs, attn_mask=None, position_ids=None,
                         plugin_input={}, **kwargs):
        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask,
                         encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs, **plugin_input)
        if hasattr(self.denoiser, 'input_feeder'):
            for feeder in self.denoiser.input_feeder:
                feeder(input_all)
        model_pred = self.denoiser(x_t, timesteps, encoder_hidden_states, encoder_attention_mask=attn_mask,
                                   added_cond_kwargs=added_cond_kwargs).sample  # Predict the noise residual
        return model_pred

    def model_forward(self, prompt_ids, image, attn_mask=None, position_ids=None, neg_prompt_ids=None, neg_attn_mask=None, neg_position_ids=None,
                      crop_info=None, plugin_input={}):
        # input prepare
        x_0 = self.get_latents(image)
        x_t, noise, timesteps = self.noise_sampler.add_noise_rand_t(x_0)
        x_t_in = x_t*self.noise_sampler.sigma_scheduler.c_in(timesteps).to(dtype=x_t.dtype).view(-1,1,1,1)
        t_in = self.noise_sampler.sigma_scheduler.c_noise(timesteps)

        if neg_prompt_ids:
            prompt_ids = self.pn_cat(neg_prompt_ids, prompt_ids)
            if neg_attn_mask:
                attn_mask = self.pn_cat(neg_attn_mask, attn_mask)
            if neg_position_ids:
                position_ids = self.pn_cat(neg_position_ids, position_ids)

        # model forward
        x_t_in, t_in = self.cfg_context.pre(x_t_in, t_in)
        encoder_hidden_states, pooled_output = self.forward_TE(prompt_ids, t_in, attn_mask=attn_mask, position_ids=position_ids,
                                                               plugin_input=plugin_input)
        added_cond_kwargs = {"text_embeds":pooled_output, "time_ids":crop_info}
        model_pred = self.forward_denoiser(x_t_in, prompt_ids, encoder_hidden_states, t_in, added_cond_kwargs=added_cond_kwargs,
                                           attn_mask=attn_mask, position_ids=position_ids, plugin_input=plugin_input)
        model_pred = self.cfg_context.post(model_pred)

        return dict(model_pred=model_pred, noise=noise, timesteps=timesteps, x_0=x_0, x_t=x_t, noise_sampler=self.noise_sampler)
