from contextlib import nullcontext

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from rainbowneko.models.wrapper import BaseWrapper
from torch import Tensor
from torch import nn
from typing import Dict

from hcpdiff.diffusion.sampler import EDM_DDPMSampler, BaseSampler, DDPMDiscreteSigmaScheduler
from hcpdiff.models import EmbeddingPTHook, TEEXHook
from hcpdiff.utils import pad_attn_bias, auto_text_encoder_cls
from hcpdiff.utils.net_utils import auto_tokenizer_cls
from .utils import TEHookCFG
from ..cfg_context import CFGContext

class StableDiffusionWrapper(BaseWrapper):
    def __init__(self, unet: UNet2DConditionModel, TE, vae: AutoencoderKL, noise_sampler: BaseSampler, tokenizer, min_attnmask=32,
                 pred_type='eps', TE_hook_cfg=TEHookCFG(), cfg_context=CFGContext(), key_map_in=None, key_map_out=None):
        super().__init__()
        self.key_mapper_in = self.build_mapper(key_map_in, None, ('prompt -> prompt_ids', 'image -> image', 'attn_mask -> attn_mask', 'position_ids -> position_ids', 'neg_prompt -> neg_prompt_ids', 'neg_attn_mask -> neg_attn_mask', 'neg_position_ids -> neg_position_ids', 'plugin_input -> plugin_input'))
        self.key_mapper_out = self.build_mapper(key_map_out, None, None)

        self.unet = unet
        self.TE = TE
        self.vae = vae
        self.noise_sampler = noise_sampler
        self.tokenizer = tokenizer
        self.min_attnmask = min_attnmask

        self.pred_type = pred_type

        self.TE_hook_cfg = TEHookCFG.create(TE_hook_cfg)
        self.cfg_context = cfg_context
        self.tokenizer.N_repeats = self.TE_hook_cfg.tokenizer_repeats

    def post_init(self):
        # Hook and extend text_encoder
        self.text_enc_hook = TEEXHook.hook(self.TE, self.tokenizer, N_repeats=self.TE_hook_cfg.tokenizer_repeats,
                                           clip_skip=self.TE_hook_cfg.clip_skip, clip_final_norm=self.TE_hook_cfg.clip_final_norm)

    def make_TE_hook(self, TE_hook_cfg):
        self.post_init()

    @property
    def vae_trainable(self):
        return False

    @property
    def TE_trainable(self):
        return False

    def get_latents(self, image: Tensor):
        if image.shape[1] == 3:
            with torch.no_grad() if self.vae_trainable else nullcontext():
                latents = self.vae.encode(image.to(dtype=self.vae.dtype)).latent_dist.sample()
                latents = latents*self.vae.config.scaling_factor
        else:
            latents = image  # Cached latents
        return latents

    def forward_TE(self, prompt_ids, timesteps, attn_mask=None, position_ids=None, plugin_input={}):
        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask, **plugin_input)
        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        # Get the text embedding for conditioning
        encoder_hidden_states = self.TE(prompt_ids, position_ids=position_ids, attention_mask=attn_mask, output_hidden_states=True)[0]

        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)
        return encoder_hidden_states, attn_mask

    def forward_unet(self, x_t, prompt_ids, encoder_hidden_states, timesteps, attn_mask=None, position_ids=None, plugin_input={}):
        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask,
                         encoder_hidden_states=encoder_hidden_states, **plugin_input)
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(input_all)
        model_pred = self.unet(x_t, timesteps, encoder_hidden_states, encoder_attention_mask=attn_mask).sample  # Predict the noise residual
        return model_pred

    def model_forward(self, prompt_ids, image, attn_mask=None, position_ids=None, neg_prompt_ids=None, neg_attn_mask=None, neg_position_ids=None,
                      plugin_input={}):
        # input prepare
        x_0 = self.get_latents(image)
        x_t, noise, sigma, timesteps = self.noise_sampler.add_noise_rand_t(x_0)
        x_t_in = x_t*self.noise_sampler.c_in(sigma).to(dtype=x_t.dtype)

        if neg_prompt_ids:
            prompt_ids = torch.cat([neg_prompt_ids, prompt_ids], dim=0)
            if neg_attn_mask:
                attn_mask = torch.cat([neg_attn_mask, attn_mask], dim=0)
            if neg_position_ids:
                position_ids = torch.cat([neg_position_ids, position_ids], dim=0)

        # model forward
        x_t_in, timesteps = self.cfg_context.pre(x_t_in, timesteps)
        encoder_hidden_states, attn_mask = self.forward_TE(prompt_ids, timesteps, attn_mask=attn_mask, position_ids=position_ids,
                                                           plugin_input=plugin_input)
        model_pred = self.forward_unet(x_t_in, prompt_ids, encoder_hidden_states, timesteps, attn_mask=attn_mask, position_ids=position_ids,
                                       plugin_input=plugin_input)
        model_pred = self.cfg_context.post(model_pred)

        return dict(model_pred=model_pred, noise=noise, sigma=sigma, timesteps=timesteps, x_0=x_0, x_t=x_t, pred_type=self.pred_type, noise_sampler=self.noise_sampler)

    def forward(self, ds_name=None, **kwargs):
        model_args, model_kwargs = self.get_map_data(self.key_mapper_in, kwargs, ds_name)
        out = self.model_forward(*model_args, **model_kwargs)
        return self.get_map_data(self.key_mapper_out, out, ds_name=ds_name)[1]

    def enable_gradient_checkpointing(self):
        def grad_ckpt_enable(m):
            if getattr(m, 'gradient_checkpointing', False):
                m.training = True

        self.unet.enable_gradient_checkpointing()
        if self.TE_trainable:
            self.TE.gradient_checkpointing_enable()
        self.apply(grad_ckpt_enable)

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
        self.vae = self.vae.to(dtype=vae_dtype)
        if not self.TE_trainable:
            self.TE = self.TE.to(dtype=dtype)

    @classmethod
    def from_pretrained(cls, pretrained_model, unet=None, TE=None, vae: AutoencoderKL = None, noise_sampler: BaseSampler = None,
                        tokenizer=None, revision=None, **kwargs):
        unet = unet or UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder="unet", revision=revision
        )
        vae = vae or AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=revision)
        noise_sampler = noise_sampler or EDM_DDPMSampler(DDPMDiscreteSigmaScheduler())

        if TE is None:
            # import correct text encoder class
            text_encoder_cls = auto_text_encoder_cls(pretrained_model, revision)
            TE = text_encoder_cls.from_pretrained(
                pretrained_model, subfolder="text_encoder", revision=revision
            )

        if tokenizer is None:
            tokenizer_cls = auto_tokenizer_cls(pretrained_model, revision)
            tokenizer = tokenizer_cls.from_pretrained(pretrained_model, subfolder="tokenizer", revision=revision, use_fast=False)

        return cls(unet, TE, vae, noise_sampler, tokenizer, **kwargs)
