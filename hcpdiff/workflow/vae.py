import torch
from diffusers.image_processor import VaeImageProcessor
from hcpdiff.utils import to_cuda, to_cpu
from hcpdiff.utils.net_utils import get_dtype
from rainbowneko.infer import BasicAction

class EncodeAction(BasicAction):
    def __init__(self, image_processor=None, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.image_processor = image_processor

    def forward(self, vae, images, dtype: str, device, generator, bs=None, model_offload=False, **states):
        if bs is None:
            if 'prompt' in states:
                bs = len(states['prompt'])
        vae_scale_factor = 2**(len(vae.config.block_out_channels)-1)
        if self.image_processor is None:
            self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

        image = self.image_processor.preprocess(images)
        if bs is not None and image.shape[0] != bs:
            image = image.repeat(bs//image.shape[0], 1, 1, 1)
        image = image.to(device=device, dtype=vae.dtype)

        if image.shape[1] == 4:
            init_latents = image
        else:
            if model_offload:
                to_cuda(vae)
            if isinstance(generator, list) and len(generator) != bs:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {bs}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    vae.encode(image[i: i+1]).latent_dist.sample(generator[i]) for i in range(bs)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = vae.encode(image).latent_dist.sample(generator)

            init_latents = init_latents.to(dtype=get_dtype(dtype))
            if shift_factor := getattr(vae.config, 'shift_factor', None) is not None:
                init_latents = (init_latents-shift_factor)*vae.config.scaling_factor
            else:
                init_latents = init_latents*vae.config.scaling_factor
            if model_offload:
                to_cpu(vae)
        return {'latents':init_latents}

class DecodeAction(BasicAction):
    def __init__(self, image_processor=None, output_type='pil', key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)

        self.image_processor = image_processor
        self.output_type = output_type

    def forward(self, vae, denoiser, latents, model_offload=False, **states):
        vae_scale_factor = 2**(len(vae.config.block_out_channels)-1)
        if self.image_processor is None:
            self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

        if model_offload:
            to_cpu(denoiser)
            torch.cuda.synchronize()
            to_cuda(vae)
        latents = latents.to(dtype=vae.dtype)
        if shift_factor := getattr(vae.config, 'shift_factor', None) is not None:
            latents = latents/vae.config.scaling_factor + shift_factor
        else:
            latents = latents/vae.config.scaling_factor
        image = vae.decode(latents, return_dict=False)[0]
        if model_offload:
            to_cpu(vae)

        do_denormalize = [True]*image.shape[0]
        image = self.image_processor.postprocess(image, output_type=self.output_type, do_denormalize=do_denormalize)
        return {'images':image}
