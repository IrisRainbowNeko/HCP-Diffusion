from .base import BasicAction, from_memory_context
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from typing import Dict, Any
import torch
from hcpdiff.utils import to_cuda, to_cpu
from hcpdiff.utils.net_utils import get_dtype

class EncodeAction(BasicAction):
    @from_memory_context
    def __init__(self, vae: AutoencoderKL, image_processor=None, offload: Dict[str, Any] = None):
        super().__init__()
        self.vae = vae
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels)-1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor) if image_processor is None else image_processor
        self.offload = offload

    def forward(self, images, dtype:str, device, generator, bs=None, **states):
        if bs is None:
            if 'prompt' in states:
                bs = len(states['prompt'])

        image = self.image_processor.preprocess(images)
        image = image.to(device=device, dtype=self.vae.dtype)

        if image.shape[1] == 4:
            init_latents = image
        else:
            if self.offload:
                to_cuda(self.vae)
            if isinstance(generator, list) and len(generator) != bs:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {bs}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(bs)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents.to(dtype=get_dtype(dtype))
            if self.offload:
                to_cpu(self.vae)
        return {**states, 'latents':init_latents, 'dtype':dtype, 'device':device, 'bs':bs}

class DecodeAction(BasicAction):
    @from_memory_context
    def __init__(self, vae: AutoencoderKL, image_processor=None, output_type='pil', offload: Dict[str, Any] = None, decode_key='latents'):
        super().__init__()
        self.vae = vae
        self.offload = offload

        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels)-1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor) if image_processor is None else image_processor
        self.output_type = output_type
        self.decode_key = decode_key

    def forward(self, **states):
        latents = states[self.decode_key]
        if self.offload:
            to_cuda(self.vae)
        latents = latents.to(dtype=self.vae.dtype)
        image = self.vae.decode(latents/self.vae.config.scaling_factor, return_dict=False)[0]
        if self.offload:
            to_cpu(self.vae)

        do_denormalize = [True]*image.shape[0]
        image = self.image_processor.postprocess(image, output_type=self.output_type, do_denormalize=do_denormalize)
        return {**states, 'images':image}
