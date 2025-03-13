import torch
from hcpdiff.ckpt_manager import DiffusersSD15Format
from rainbowneko.ckpt_manager import ModelManager, LocalCkptSource

def sd15_auto_loader(ckpt_path, unet=None, TE=None, vae=None, noise_sampler=None,
                     tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
    manager = ModelManager(
        format=DiffusersSD15Format(),
        source=LocalCkptSource(),
    )
    models = manager.load(ckpt_path, unet=unet, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer, revision=revision,
                          dtype=dtype, **kwargs)
    return models
