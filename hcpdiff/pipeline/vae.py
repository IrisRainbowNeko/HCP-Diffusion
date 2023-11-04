from .base import BasicAction, from_memory_context
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from typing import Dict, Any
import torch

class EncodeAction(BasicAction):
    @from_memory_context
    def __init__(self, vae: AutoencoderKL = None, offload: Dict[str, Any] = None):
        super().__init__()
        self.vae = vae
        self.offload = offload

    def forward(self, image, dtype, device, generator, batch_size, **states):
        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            init_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents
        return {'latents':init_latents, 'dtype':dtype, 'device':device, 'batch_size':batch_size, **states}

class DecodeAction(BasicAction):
    @from_memory_context
    def __init__(self, vae: AutoencoderKL = None, image_processor=None, output_type='pil', offload: Dict[str, Any] = None):
        super().__init__()
        self.vae = vae
        self.offload = offload

        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels)-1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor) if image_processor is None else image_processor
        self.output_type = output_type

    def forward(self, latents, **states):
        latents = latents.to(dtype=self.vae.dtype)
        image = self.vae.decode(latents/self.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True]*image.shape[0]
        image = self.image_processor.postprocess(image, output_type=self.output_type, do_denormalize=do_denormalize)
        return {'image':image, **states}