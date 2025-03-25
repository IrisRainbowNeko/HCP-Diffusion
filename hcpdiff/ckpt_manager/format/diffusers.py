import torch
from diffusers import ModelMixin, AutoencoderKL, UNet2DConditionModel, PixArtTransformer2DModel
from rainbowneko.ckpt_manager.format import CkptFormat
from transformers import CLIPTextModel, AutoTokenizer, T5EncoderModel

from hcpdiff.diffusion.sampler import DDPMSampler, DDPMDiscreteSigmaScheduler
from hcpdiff.models.compose import SDXLTokenizer, SDXLTextEncoder

class DiffusersModelFormat(CkptFormat):
    def __init__(self, builder: ModelMixin):
        self.builder = builder

    def save_ckpt(self, sd_model: ModelMixin, save_f: str, **kwargs):
        sd_model.save_pretrained(save_f)

    def load_ckpt(self, ckpt_f: str, map_location="cpu", **kwargs):
        self.builder.from_pretrained(ckpt_f, **kwargs)

class DiffusersSD15Format(CkptFormat):
    def load_ckpt(self, pretrained_model: str, map_location="cpu", denoiser=None, TE=None, vae: AutoencoderKL = None, noise_sampler=None,
                  tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
        denoiser = denoiser or UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder="unet", revision=revision, torch_dtype=dtype
        )
        vae = vae or AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=revision, torch_dtype=dtype)
        noise_sampler = noise_sampler or DDPMSampler(DDPMDiscreteSigmaScheduler())

        TE = TE or CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder", revision=revision, torch_dtype=dtype)
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer", revision=revision, use_fast=False)

        return dict(denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer)

class DiffusersSDXLFormat(CkptFormat):
    def load_ckpt(self, pretrained_model: str, map_location="cpu", denoiser=None, TE=None, vae: AutoencoderKL = None, noise_sampler=None,
                  tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
        denoiser = denoiser or UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder="unet", revision=revision, torch_dtype=dtype
        )
        vae = vae or AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=revision, torch_dtype=dtype)
        noise_sampler = noise_sampler or DDPMSampler(DDPMDiscreteSigmaScheduler())

        TE = TE or SDXLTextEncoder.from_pretrained(pretrained_model, subfolder="text_encoder", revision=revision, torch_dtype=dtype)
        tokenizer = tokenizer or SDXLTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer", revision=revision, use_fast=False)

        return dict(denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer)

class DiffusersPixArtFormat(CkptFormat):
    def load_ckpt(self, pretrained_model: str, map_location="cpu", denoiser=None, TE=None, vae: AutoencoderKL = None, noise_sampler=None,
                  tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
        denoiser = denoiser or PixArtTransformer2DModel.from_pretrained(
            pretrained_model, subfolder="transformer", revision=revision, torch_dtype=dtype
        )
        vae = vae or AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=revision, torch_dtype=dtype)
        noise_sampler = noise_sampler or DDPMSampler(DDPMDiscreteSigmaScheduler())

        TE = TE or T5EncoderModel.from_pretrained(pretrained_model, subfolder="text_encoder", revision=revision, torch_dtype=dtype)
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer", revision=revision, use_fast=False)

        return dict(denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer)
