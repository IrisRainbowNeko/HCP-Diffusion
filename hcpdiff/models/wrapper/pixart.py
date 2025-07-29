from .sd import SD15Wrapper
from hcpdiff.utils import pad_attn_bias

class PixArtWrapper(SD15Wrapper):
    def forward_TE(self, prompt_ids, timesteps, attn_mask=None, plugin_input={}, **kwargs):
        # T5Encoder do not need position_ids (It use relative position embedding for key and query)
        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, attn_mask=attn_mask, **plugin_input)
        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        # Get the text embedding for conditioning
        encoder_hidden_states = self.TE(prompt_ids, attention_mask=attn_mask, output_hidden_states=True)[0]
        return encoder_hidden_states
        
    def forward_denoiser(self, x_t, prompt_ids, encoder_hidden_states, timesteps, attn_mask=None, position_ids=None, resolution=None, aspect_ratio=None,
                     plugin_input={}, **kwargs):
        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask,
                         encoder_hidden_states=encoder_hidden_states, **plugin_input)
        if hasattr(self.denoiser, 'input_feeder'):
            for feeder in self.denoiser.input_feeder:
                feeder(input_all)
        added_cond_kwargs = {"resolution":resolution, "aspect_ratio":aspect_ratio}
        model_pred = self.denoiser(x_t, encoder_hidden_states, timesteps, encoder_attention_mask=attn_mask,
                               added_cond_kwargs=added_cond_kwargs).sample  # Predict the noise residual

        # remove pred vars for pixart output (see DiT for more)
        return model_pred.chunk(2, dim=1)[0]
