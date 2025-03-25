import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, StableDiffusionXLPipeline
from rainbowneko.ckpt_manager.format import CkptFormat

from hcpdiff.diffusion.sampler import DDPMSampler, DDPMDiscreteSigmaScheduler
from hcpdiff.models.compose import SDXLTextEncoder, SDXLTokenizer

class OfficialSD15Format(CkptFormat):
    # Single file format
    def load_ckpt(self, pretrained_model: str, map_location="cpu", denoiser=None, TE=None, vae: AutoencoderKL = None, noise_sampler=None,
                  tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
        pipe = StableDiffusionPipeline.from_single_file(
            pretrained_model, revision=revision, torch_dtype=dtype, unet=denoiser, vae=vae, text_encoder=TE, tokenizer=tokenizer
        )
        noise_sampler = noise_sampler or DDPMSampler(DDPMDiscreteSigmaScheduler())
        return dict(denoiser=pipe.unet, TE=pipe.text_encoder, vae=pipe.vae, noise_sampler=noise_sampler, tokenizer=pipe.tokenizer)

class OfficialSDXLFormat(CkptFormat):
    # Single file format
    def load_ckpt(self, pretrained_model: str, map_location="cpu", denoiser=None, TE=None, vae: AutoencoderKL = None, noise_sampler=None,
                  tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
        if TE is None:
            pipe = StableDiffusionXLPipeline.from_single_file(
                pretrained_model, revision=revision, torch_dtype=dtype, unet=denoiser, vae=vae
            )
        else:
            pipe = StableDiffusionXLPipeline.from_single_file(
                pretrained_model, revision=revision, torch_dtype=dtype, unet=denoiser, vae=vae,
                text_encoder=TE.clip_L, text_encoder_2=TE.clip_bigG, tokenizer=tokenizer.clip_L, tokenizer_2=tokenizer.clip_bigG,
            )

        noise_sampler = noise_sampler or DDPMSampler(DDPMDiscreteSigmaScheduler())
        TE = SDXLTextEncoder([('clip_L', pipe.text_encoder), ('clip_bigG', pipe.text_encoder_2)])
        tokenizer = SDXLTokenizer([('clip_L', pipe.tokenizer), ('clip_bigG', pipe.tokenizer_2)])

        return dict(denoiser=pipe.unet, TE=TE, vae=pipe.vae, noise_sampler=noise_sampler, tokenizer=tokenizer)
