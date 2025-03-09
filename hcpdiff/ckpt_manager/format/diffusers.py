from diffusers import ModelMixin, AutoencoderKL, UNet2DConditionModel
from hcpdiff.diffusion.sampler import EDM_DDPMSampler, DDPMDiscreteSigmaScheduler
from hcpdiff.utils import auto_tokenizer_cls, auto_text_encoder_cls
from rainbowneko.ckpt_manager.format import CkptFormat
import torch

class DiffusersModelFormat(CkptFormat):
    def __init__(self, builder: ModelMixin):
        self.builder = builder

    def save_ckpt(self, sd_model: ModelMixin, save_f: str, **kwargs):
        sd_model.save_pretrained(save_f)

    def load_ckpt(self, ckpt_f: str, map_location="cpu", **kwargs):
        self.builder.from_pretrained(ckpt_f, **kwargs)


class DiffusersSD15Format(CkptFormat):
    def load_ckpt(self, pretrained_model: str, map_location="cpu", unet=None, TE=None, vae: AutoencoderKL = None, noise_sampler = None,
                        tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
        unet = unet or UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder="unet", revision=revision, torch_dtype=dtype
        )
        vae = vae or AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=revision, torch_dtype=dtype)
        noise_sampler = noise_sampler or EDM_DDPMSampler(DDPMDiscreteSigmaScheduler())

        if TE is None:
            # import correct text encoder class
            text_encoder_cls = auto_text_encoder_cls(pretrained_model, revision)
            TE = text_encoder_cls.from_pretrained(
                pretrained_model, subfolder="text_encoder", revision=revision, torch_dtype=dtype
            )

        if tokenizer is None:
            tokenizer_cls = auto_tokenizer_cls(pretrained_model, revision)
            tokenizer = tokenizer_cls.from_pretrained(pretrained_model, subfolder="tokenizer", revision=revision, use_fast=False)

        return dict(unet=unet, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer)