from torch import nn
import itertools
from transformers import CLIPTextModel
from hcpdiff.utils import pad_attn_bias

class TEUnetWrapper(nn.Module):
    def __init__(self, unet, TE, train_TE=False):
        super().__init__()
        self.unet = unet
        self.TE = TE

        self.train_TE = train_TE

    def forward(self, prompt_ids, noisy_latents, timesteps, attn_mask=None, position_ids=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask, **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        encoder_hidden_states = self.TE(prompt_ids, position_ids=position_ids, attention_mask=attn_mask, output_hidden_states=True)[0]  # Get the text embedding for conditioning

        if attn_mask is not None:
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all['encoder_hidden_states'] = encoder_hidden_states
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(input_all)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, encoder_attention_mask=attn_mask).sample  # Predict the noise residual
        return model_pred

    def prepare(self, accelerator):
        if self.train_TE:
            return accelerator.prepare(self)
        else:
            self.unet = accelerator.prepare(self.unet)
            return self

    def enable_gradient_checkpointing(self):
        def grad_ckpt_enable(m):
            if hasattr(m, 'gradient_checkpointing'):
                m.training = True

        self.unet.enable_gradient_checkpointing()
        if self.train_TE:
            self.TE.gradient_checkpointing_enable()
            self.apply(grad_ckpt_enable)
        else:
            self.unet.apply(grad_ckpt_enable)

    def trainable_parameters(self):
        if self.train_TE:
            return itertools.chain(self.unet.parameters(), self.TE.parameters())
        else:
            return self.unet.parameters()

class SDXLTEUnetWrapper(TEUnetWrapper):
    def forward(self, prompt_ids, noisy_latents, timesteps, attn_mask=None, position_ids=None, crop_info=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask, **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        encoder_hidden_states, pooled_output = self.TE(prompt_ids, position_ids=position_ids, attention_mask=attn_mask, output_hidden_states=True)  # Get the text embedding for conditioning

        added_cond_kwargs = {"text_embeds":pooled_output[-1], "time_ids":crop_info}
        if attn_mask is not None:
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all['encoder_hidden_states'] = encoder_hidden_states
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(input_all)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, encoder_attention_mask=attn_mask, added_cond_kwargs=added_cond_kwargs).sample  # Predict the noise residual
        return model_pred