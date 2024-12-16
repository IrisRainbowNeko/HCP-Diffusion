from torch import nn
import itertools
from transformers import CLIPTextModel, T5EncoderModel
from hcpdiff.utils import pad_attn_bias, auto_text_encoder_cls
from hcpdiff.models.compose import SDXLTextEncoder
from diffusers import UNet2DConditionModel
from torch.nn.parallel.distributed import DistributedDataParallel
from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel

class TEUnetWrapper(nn.Module):
    def __init__(self, unet, TE, train_TE=False, min_attnmask=32):
        super().__init__()
        self.unet = unet
        self.TE = TE

        self.train_TE = train_TE
        self.min_attnmask = min_attnmask

    def forward(self, prompt_ids, noisy_latents, timesteps, attn_mask=None, position_ids=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask,
                         **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        encoder_hidden_states = self.TE(prompt_ids, position_ids=position_ids, attention_mask=attn_mask, output_hidden_states=True)[0]  # Get the text embedding for conditioning

        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all['encoder_hidden_states'] = encoder_hidden_states
        if isinstance(self.unet, DistributedDataParallel):
            if hasattr(self.unet.module, 'input_feeder'):
                for feeder in self.unet.module.input_feeder:
                    feeder(input_all)
        else:
            if hasattr(self.unet, 'input_feeder'):
                for feeder in self.unet.input_feeder:
                    feeder(input_all)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, encoder_attention_mask=attn_mask).sample  # Predict the noise residual
        return model_pred

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

    @classmethod
    def build_from_pretrained(cls, pretrained_model_name_or_path, unet=None, TE=None, revision=None, train_TE=False, min_attnmask=32):
        unet = unet or UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", revision=revision
        )

        if TE is None:
            # import correct text encoder class
            text_encoder_cls = auto_text_encoder_cls(pretrained_model_name_or_path, revision)
            TE = text_encoder_cls.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
            )

        return cls(unet, TE, train_TE, min_attnmask)

class SDXLTEUnetWrapper(TEUnetWrapper):
    def forward(self, prompt_ids, noisy_latents, timesteps, attn_mask=None, position_ids=None, crop_info=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask,
                         **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        encoder_hidden_states, pooled_output = self.TE(prompt_ids, position_ids=position_ids, attention_mask=attn_mask,
                                                       output_hidden_states=True)  # Get the text embedding for conditioning

        added_cond_kwargs = {"text_embeds":pooled_output[-1], "time_ids":crop_info}
        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all['encoder_hidden_states'] = encoder_hidden_states
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(input_all)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, encoder_attention_mask=attn_mask,
                               added_cond_kwargs=added_cond_kwargs).sample  # Predict the noise residual
        return model_pred

class PixArtWrapper(TEUnetWrapper):
    def __init__(self, unet, TE, train_TE=False):
        super().__init__(unet, TE, train_TE, min_attnmask=0)

    def forward(self, prompt_ids, noisy_latents, timesteps, attn_mask=None, position_ids=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask,
                         **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        encoder_hidden_states = self.TE(prompt_ids, position_ids=position_ids, attention_mask=attn_mask, output_hidden_states=True)[0]  # Get the text embedding for conditioning

        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all['encoder_hidden_states'] = encoder_hidden_states
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(input_all)
        model_pred = self.unet(noisy_latents, encoder_hidden_states, timesteps, encoder_attention_mask=attn_mask).sample  # Predict the noise residual
        return model_pred

    @classmethod
    def build_from_pretrained(cls, pretrained_model_name_or_path, unet=None, TE=None, revision=None, train_TE=False, min_attnmask=32):
        unet = unet or PixArtTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer", revision=revision
        )

        if TE is None:
            TE = T5EncoderModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
            )

        return cls(unet, TE, train_TE)

def auto_build_wrapper(pretrained_model_name_or_path, unet=None, TE=None, revision=None, train_TE=False, min_attnmask=32):
    if TE is not None:
        text_encoder_cls = type(TE)
    else:
        text_encoder_cls = auto_text_encoder_cls(pretrained_model_name_or_path, revision)

    if text_encoder_cls == SDXLTextEncoder:
        wrapper_cls = SDXLTEUnetWrapper
    else:
        wrapper_cls = TEUnetWrapper

    return wrapper_cls.build_from_pretrained(pretrained_model_name_or_path, unet, TE, revision, train_TE, min_attnmask)
