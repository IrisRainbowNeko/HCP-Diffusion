import torch
from diffusers import FluxTransformer2DModel, AutoencoderKL
from einops import repeat, rearrange
from hcpdiff.diffusion.sampler import BaseSampler
from hcpdiff.utils import pad_attn_bias
from rainbowneko.utils import add_dims

from .sd import SD15Wrapper
from .utils import TEHookCFG, SDXL_TEHookCFG
from ..cfg_context import CFGContext

class FluxWrapper(SD15Wrapper):
    def __init__(self, denoiser: FluxTransformer2DModel, TE, vae: AutoencoderKL, noise_sampler: BaseSampler, tokenizer, min_attnmask=0,
                 guidance=5.0, patch_size=2, TE_hook_cfg: TEHookCFG = SDXL_TEHookCFG, cfg_context=CFGContext(), key_map_in=None, key_map_out=None):
        super().__init__(denoiser, TE, vae, noise_sampler, tokenizer, min_attnmask, TE_hook_cfg, cfg_context, key_map_in, key_map_out)
        self.key_mapper_in = self.build_mapper(key_map_in, None, (
            'prompt -> prompt_ids', 'image -> image', 'attn_mask -> attn_mask', 'neg_prompt -> neg_prompt_ids',
            'neg_attn_mask -> neg_attn_mask', 'plugin_input -> plugin_input'))
        self.guidance = guidance
        self.patch_size = patch_size

    def forward_TE(self, prompt_ids, timesteps, attn_mask=None, position_ids=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask, **plugin_input)
        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        # Get the text embedding for conditioning
        encoder_hidden_states, pooled_output = self.TE(prompt_ids, position_ids=position_ids, attention_mask=attn_mask, output_hidden_states=True)
        return encoder_hidden_states, pooled_output

    def forward_denoiser(self, x_t, H, W, prompt_ids, encoder_hidden_states, pooled_output, timesteps, attn_mask=None, plugin_input={}, **kwargs):
        attn_mask = attn_mask['T5']
        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        img_ids = torch.zeros(H, W, 3)
        img_ids[..., 1] = img_ids[..., 1]+torch.arange(H)[:, None]
        img_ids[..., 2] = img_ids[..., 2]+torch.arange(W)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=x_t.shape[0])

        txt_ids = torch.zeros(x_t.shape[0], encoder_hidden_states.shape[1], 3)

        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, attn_mask=attn_mask, img_ids=img_ids, txt_ids=txt_ids,
                         encoder_hidden_states=encoder_hidden_states, **plugin_input)
        if hasattr(self.denoiser, 'input_feeder'):
            for feeder in self.denoiser.input_feeder:
                feeder(input_all)
        model_pred = self.denoiser(x_t, timesteps, self.guidance, pooled_output, encoder_hidden_states, img_ids=img_ids, txt_ids=txt_ids).sample
        return model_pred

    def model_forward(self, prompt_ids, image, attn_mask=None, neg_prompt_ids=None, neg_attn_mask=None, plugin_input={}, **kwargs):
        # input prepare
        x_0 = self.get_latents(image)
        B, C, H, W = x_0.shape
        x_0_patch = rearrange(x_0, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=self.patch_size, pw=self.patch_size)
        x_t, noise, timesteps = self.noise_sampler.add_noise_rand_t(x_0_patch)
        x_t_in = x_t*add_dims(self.noise_sampler.sigma_scheduler.c_in(timesteps).to(dtype=x_t.dtype), x_t.ndim-1)
        t_in = self.noise_sampler.sigma_scheduler.c_noise(timesteps)
        x_t_in = rearrange(x_t_in, "b c h w -> b (h w) c")

        if neg_prompt_ids:
            prompt_ids = self.pn_cat(neg_prompt_ids, prompt_ids)
            if neg_attn_mask:
                attn_mask = self.pn_cat(neg_attn_mask, attn_mask)

        # model forward
        x_t_in, t_in = self.cfg_context.pre(x_t_in, t_in)
        encoder_hidden_states, pooled_output = self.forward_TE(prompt_ids, t_in, attn_mask=attn_mask, plugin_input=plugin_input, **kwargs)
        model_pred = self.forward_denoiser(x_t_in, H, W, prompt_ids, encoder_hidden_states, pooled_output, t_in, attn_mask=attn_mask,
                                           plugin_input=plugin_input, **kwargs)
        model_pred = rearrange(model_pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=self.patch_size, pw=self.patch_size, h=H, w=W)
        model_pred = self.cfg_context.post(model_pred)

        return dict(model_pred=model_pred, noise=noise, timesteps=timesteps, x_0=x_0, x_t=x_t, noise_sampler=self.noise_sampler)
