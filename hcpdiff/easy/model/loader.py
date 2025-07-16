import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline
from rainbowneko.ckpt_manager import NekoLoader, LocalCkptSource

from hcpdiff.ckpt_manager import DiffusersSD15Format, DiffusersSDXLFormat, DiffusersPixArtFormat, OfficialSD15Format, OfficialSDXLFormat, \
    DiffusersFluxFormat, OneFileFluxFormat
from hcpdiff.models.compose import SDXLTextEncoder, FluxTextEncoder
from hcpdiff.models.wrapper import SDXLWrapper, SD15Wrapper, PixArtWrapper, FluxWrapper
from hcpdiff.utils import auto_text_encoder_cls, get_pipe_name

def SD15_auto_loader(ckpt_path, denoiser=None, TE=None, vae=None, noise_sampler=None,
                     tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
    try:
        try_diffusers = StableDiffusionPipeline.load_config(ckpt_path)
        loader = NekoLoader(
            format=DiffusersSD15Format(),
            source=LocalCkptSource(),
        )
    except EnvironmentError:
        loader = NekoLoader(
            format=OfficialSD15Format(),
            source=LocalCkptSource(),
        )
    models = loader.load(ckpt_path, denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer, revision=revision,
                         dtype=dtype, **kwargs)
    return models

def SDXL_auto_loader(ckpt_path, denoiser=None, TE=None, vae=None, noise_sampler=None,
                     tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
    try:
        try_diffusers = StableDiffusionXLPipeline.load_config(ckpt_path)
        loader = NekoLoader(
            format=DiffusersSDXLFormat(),
            source=LocalCkptSource(),
        )
    except EnvironmentError:
        loader = NekoLoader(
            format=OfficialSDXLFormat(),
            source=LocalCkptSource(),
        )
    models = loader.load(ckpt_path, denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer, revision=revision,
                         dtype=dtype, **kwargs)
    return models

def PixArt_auto_loader(ckpt_path, denoiser=None, TE=None, vae=None, noise_sampler=None,
                       tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
    loader = NekoLoader(
        format=DiffusersPixArtFormat(),
        source=LocalCkptSource(),
    )
    models = loader.load(ckpt_path, denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer, revision=revision,
                         dtype=dtype, **kwargs)
    return models

def Flux_auto_loader(ckpt_path, denoiser=None, TE=None, vae=None, noise_sampler=None,
                     tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
    try:
        try_diffusers = FluxPipeline.load_config(ckpt_path)
        loader = NekoLoader(
            format=DiffusersFluxFormat(),
            source=LocalCkptSource(),
        )
    except EnvironmentError:
        loader = NekoLoader(
            format=OneFileFluxFormat(),
            source=LocalCkptSource(),
        )
    models = loader.load(ckpt_path, denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer, revision=revision,
                         dtype=dtype, **kwargs)
    return models

def auto_load_wrapper(pretrained_model, denoiser=None, TE=None, vae=None, noise_sampler=None, tokenizer=None, revision=None,
                      dtype=torch.float32, **kwargs):
    if TE is not None:
        text_encoder_cls = type(TE)
    else:
        text_encoder_cls = auto_text_encoder_cls(pretrained_model, revision)

    pipe_name = get_pipe_name(pretrained_model)

    if text_encoder_cls == SDXLTextEncoder:
        wrapper_cls = SDXLWrapper
        format = DiffusersSDXLFormat()
    elif text_encoder_cls == FluxTextEncoder:
        wrapper_cls = FluxWrapper
        format = DiffusersFluxFormat()
    elif 'PixArt' in pipe_name:
        wrapper_cls = PixArtWrapper
        format = DiffusersPixArtFormat()
    else:
        wrapper_cls = SD15Wrapper
        format = DiffusersSD15Format()

    loader = NekoLoader(
        format=format,
        source=LocalCkptSource(),
    )
    models = loader.load(pretrained_model, denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer, revision=revision,
                         dtype=dtype)

    return wrapper_cls.build_from_pretrained(models, **kwargs)
