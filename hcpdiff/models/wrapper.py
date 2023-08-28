from torch import nn
import itertools

class TEUnetWrapper(nn.Module):
    def __init__(self, unet, TE, train_TE=False):
        super().__init__()
        self.unet = unet
        self.TE = TE

        self.train_TE = train_TE

    def forward(self, prompt_ids, noisy_latents, timesteps, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps, **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(input_all)

        encoder_hidden_states = self.TE(prompt_ids, output_hidden_states=True)[0]  # Get the text embedding for conditioning
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample  # Predict the noise residual
        return model_pred

    def prepare(self, accelerator):
        if self.train_TE:
            return accelerator.prepare(self)
        else:
            self.unet = accelerator.prepare(self.unet)
            return self

    def freeze_model(self):
        if self.train_TE:
            for name, m in self.named_modules():
                if isinstance(m, nn.Dropout) and not m.training:
                    m.p = 0.
            self.train()
        else:
            for name, m in self.unet.named_modules():
                if isinstance(m, nn.Dropout) and not m.training:
                    m.p = 0.
            self.unet.train()

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()
        if self.train_TE:
            self.TE.gradient_checkpointing_enable()

    def trainable_parameters(self):
        if self.train_TE:
            return itertools.chain(self.unet.parameters(), self.TE.parameters())
        else:
            self.unet.parameters()

class SDXLTEUnetWrapper(TEUnetWrapper):
    def forward(self, prompt_ids, noisy_latents, timesteps, crop_info=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps, **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(input_all)

        encoder_hidden_states, pooled_output = self.TE(prompt_ids, output_hidden_states=True)  # Get the text embedding for conditioning

        added_cond_kwargs = {"text_embeds":pooled_output[-1], "time_ids":crop_info}

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample  # Predict the noise residual
        return model_pred