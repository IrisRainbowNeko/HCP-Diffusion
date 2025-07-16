import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline
from rainbowneko.ckpt_manager.format import CkptFormat

from hcpdiff.diffusion.sampler import VPSampler, DDPMDiscreteSigmaScheduler, FlowSigmaScheduler, Sampler, FluxShiftTimeSampler
from hcpdiff.models.compose import SDXLTextEncoder, SDXLTokenizer, FluxTextEncoder, FluxTokenizer

class OfficialSD15Format(CkptFormat):
    # Single file format
    def load_ckpt(self, pretrained_model: str, map_location="cpu", denoiser=None, TE=None, vae: AutoencoderKL = None, noise_sampler=None,
                  tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
        pipe_args = dict(unet=denoiser, vae=vae, text_encoder=TE, tokenizer=tokenizer)
        pipe_args = {k:v for k,v in pipe_args.items() if v is not None}
        pipe = StableDiffusionPipeline.from_single_file(
            pretrained_model, revision=revision, torch_dtype=dtype, **pipe_args
        )
        noise_sampler = noise_sampler or VPSampler(DDPMDiscreteSigmaScheduler())
        return dict(denoiser=pipe.unet, TE=pipe.text_encoder, vae=pipe.vae, noise_sampler=noise_sampler, tokenizer=pipe.tokenizer)

class OfficialSDXLFormat(CkptFormat):
    # Single file format
    def load_ckpt(self, pretrained_model: str, map_location="cpu", denoiser=None, TE=None, vae: AutoencoderKL = None, noise_sampler=None,
                  tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
        pipe_args = dict(unet=denoiser, vae=vae)
        if TE is not None:
            pipe_args['text_encoder'] = TE.clip_L
            pipe_args['text_encoder_2'] = TE.clip_bigG
        if tokenizer is not None:
            pipe_args['tokenizer'] = tokenizer.clip_L
            pipe_args['tokenizer_2'] = tokenizer.clip_bigG

        pipe_args = {k:v for k,v in pipe_args.items() if v is not None}
        pipe = StableDiffusionXLPipeline.from_single_file(
            pretrained_model, revision=revision, torch_dtype=dtype, **pipe_args
        )

        noise_sampler = noise_sampler or VPSampler(DDPMDiscreteSigmaScheduler())
        TE = SDXLTextEncoder({'clip_L': pipe.text_encoder, 'clip_bigG': pipe.text_encoder_2})
        tokenizer = SDXLTokenizer({'clip_L': pipe.tokenizer, 'clip_bigG': pipe.tokenizer_2})

        return dict(denoiser=pipe.unet, TE=TE, vae=pipe.vae, noise_sampler=noise_sampler, tokenizer=tokenizer)

class OneFileFluxFormat(CkptFormat):
    # Single file format
    def load_ckpt(self, pretrained_model: str, map_location="cpu", denoiser=None, TE=None, vae: AutoencoderKL = None, noise_sampler=None,
                  tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
        pipe_args = dict(unet=denoiser, vae=vae)
        if TE is not None:
            pipe_args['text_encoder'] = TE.clip
            pipe_args['text_encoder_2'] = TE.T5
        if tokenizer is not None:
            pipe_args['tokenizer'] = tokenizer.clip
            pipe_args['tokenizer_2'] = tokenizer.T5

        pipe_args = {k:v for k,v in pipe_args.items() if v is not None}
        pipe = FluxPipeline.from_single_file(
            pretrained_model, revision=revision, torch_dtype=dtype, **pipe_args
        )

        noise_sampler = noise_sampler or Sampler(FlowSigmaScheduler(), t_sampler=FluxShiftTimeSampler())
        TE = FluxTextEncoder({'clip': pipe.text_encoder, 'T5': pipe.text_encoder_2})
        tokenizer = FluxTokenizer({'clip': pipe.tokenizer, 'T5': pipe.tokenizer_2})

        return dict(denoiser=pipe.unet, TE=TE, vae=pipe.vae, noise_sampler=noise_sampler, tokenizer=tokenizer)