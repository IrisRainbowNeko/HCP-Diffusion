import argparse
import os
import sys

import hydra
import numpy as np
import torch
from PIL import Image
from accelerate import infer_auto_device_map, dispatch_model
from diffusers import UNet2DConditionModel
from diffusers.utils import PIL_INTERPOLATION
from diffusers.utils.import_utils import is_xformers_available
from hcpdiff.models import EmbeddingPTHook, TEEXHook, TokenizerHook, LoraBlock
from hcpdiff.utils.cfg_net_tools import load_hcpdiff, make_plugin
from hcpdiff.utils.net_utils import to_cpu, to_cuda
from hcpdiff.utils.pipe_hook import HookPipe_T2I, HookPipe_I2I
from hcpdiff.utils.utils import load_config_with_cli, load_config, size_to_int, int_to_size
from torch.cuda.amp import autocast

class UnetHook():  # for controlnet
    def __init__(self, unet):
        self.unet = unet
        self.call_raw = UNet2DConditionModel.__call__
        UNet2DConditionModel.__call__ = self.unet_call

    def unet_call(self, sample, timestep, encoder_hidden_states, **kwargs):
        return self.call_raw(self.unet, sample, timestep, encoder_hidden_states)

class Visualizer:
    dtype_dict = {'fp32':torch.float32, 'amp':torch.float32, 'fp16':torch.float16}

    def __init__(self, cfgs):
        self.cfgs_raw = cfgs
        self.cfgs = hydra.utils.instantiate(self.cfgs_raw)
        self.cfg_merge = self.cfgs.merge
        self.offload = 'offload' in self.cfgs and self.cfgs.offload is not None
        self.dtype = self.dtype_dict[self.cfgs.dtype]

        self.need_inter_imgs = any(item.need_inter_imgs for item in self.cfgs.interface)

        pipeline = self.get_pipeline()
        comp = pipeline.from_pretrained(self.cfgs.pretrained_model, safety_checker=None, requires_safety_checker=False,
                                        torch_dtype=self.dtype).components
        comp.update(self.cfgs.new_components)
        self.pipe = pipeline(**comp)

        if self.cfg_merge:
            self.merge_model()

        self.pipe = self.pipe.to(torch_dtype=self.dtype)

        if 'save_model' in self.cfgs and self.cfgs.save_model is not None:
            self.save_model(self.cfgs.save_model)
            os._exit(0)

        if self.offload:
            self.build_offload(self.cfgs.offload)
        else:
            self.pipe.unet.to('cuda')
            if self.need_inter_imgs:
                self.pipe.vae.to('cuda')
            else:
                def decode_latents_offload(latents, decode_latents_raw=self.pipe.decode_latents):
                    to_cpu(self.pipe.unet)

                    self.pipe.vae.to('cuda')
                    res = decode_latents_raw(latents)
                    to_cpu(self.pipe.vae)

                    self.pipe.unet.to('cuda')
                    return res

                self.pipe.decode_latents = decode_latents_offload

        emb, _ = EmbeddingPTHook.hook_from_dir(self.cfgs.emb_dir, self.pipe.tokenizer, self.pipe.text_encoder, N_repeats=self.cfgs.N_repeats)
        self.te_hook = TEEXHook.hook_pipe(self.pipe, N_repeats=self.cfgs.N_repeats, clip_skip=self.cfgs.clip_skip)
        self.token_ex = TokenizerHook(self.pipe.tokenizer)
        UnetHook(self.pipe.unet)

        if is_xformers_available():
            self.pipe.unet.enable_xformers_memory_efficient_attention()
            # self.te_hook.enable_xformers()

    def save_model(self, save_cfg):
        for k, v in self.pipe.unet.named_modules():
            if isinstance(v, LoraBlock):
                v.reparameterization_to_host()
                v.remove()
        for k, v in self.pipe.text_encoder.named_modules():
            if isinstance(v, LoraBlock):
                v.reparameterization_to_host()
                v.remove()

        self.pipe.save_pretrained(save_cfg.path, safe_serialization=save_cfg.to_safetensors)

    def get_pipeline(self):
        if self.cfgs.condition is None:
            pipe_cls = HookPipe_T2I
        else:
            if self.cfgs.condition.type == 'i2i':
                pipe_cls = HookPipe_I2I
            elif self.cfgs.condition.type == 'controlnet':
                pipe_cls = HookPipe_T2I
            else:
                raise NotImplementedError(f'No condition type named {self.cfgs.condition.type}')

        return pipe_cls

    def build_offload(self, offload_cfg):
        vram = size_to_int(offload_cfg.max_VRAM)
        device_map = infer_auto_device_map(self.pipe.unet, max_memory={0:int_to_size(vram >> 1), "cpu":offload_cfg.max_RAM}, dtype=self.dtype)
        self.pipe.unet = dispatch_model(self.pipe.unet, device_map)
        if not offload_cfg.vae_cpu:
            device_map = infer_auto_device_map(self.pipe.vae, max_memory={0:int_to_size(vram >> 5), "cpu":offload_cfg.max_RAM}, dtype=self.dtype)
            self.pipe.vae = dispatch_model(self.pipe.vae, device_map)

        def decode_latents_offload(latents, decode_latents_raw=self.pipe.decode_latents):
            to_cpu(self.pipe.unet)

            if offload_cfg.vae_cpu:
                self.pipe.vae.to(dtype=torch.float32)
                res = decode_latents_raw(latents.cpu().to(dtype=torch.float32))
            else:
                to_cuda(self.pipe.vae)
                res = decode_latents_raw(latents)

            to_cpu(self.pipe.vae)
            to_cuda(self.pipe.unet)
            return res

        self.pipe.decode_latents = decode_latents_offload

    def merge_model(self):
        if 'plugin_cfg' in self.cfg_merge:  # Build plugins
            plugin_cfg = hydra.utils.instantiate(load_config(self.cfg_merge.plugin_cfg))
            make_plugin(self.pipe.unet, plugin_cfg.plugin_unet)
            make_plugin(self.pipe.text_encoder, plugin_cfg.plugin_TE)

        for cfg_group in self.cfg_merge.values():
            if hasattr(cfg_group, 'type'):
                if cfg_group.type == 'unet':
                    load_hcpdiff(self.pipe.unet, cfg_group)
                elif cfg_group.type == 'TE':
                    load_hcpdiff(self.pipe.text_encoder, cfg_group)

    def set_scheduler(self, scheduler):
        self.pipe.scheduler = scheduler

    def prepare_cond_image(self, image, width, height, batch_size, device):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image[0], Image.Image):
                image = [
                    np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32)/255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image = image.repeat_interleave(batch_size, dim=0)
        image = image.to(device=device)

        return image

    def get_ex_input(self):
        ex_input_dict = {}
        if self.cfgs.condition is not None:
            img = Image.open(self.cfgs.condition.image).convert('RGB')
            ex_input_dict['cond'] = self.prepare_cond_image(img, self.cfgs.infer_args.width, self.cfgs.infer_args.height, self.cfgs.bs*2, 'cuda')
        return ex_input_dict

    @torch.no_grad()
    def vis_images(self, prompt, negative_prompt='', **kwargs):
        ex_input_dict = self.get_ex_input()

        to_cuda(self.pipe.text_encoder)

        mult_p, clean_text_p = self.token_ex.parse_attn_mult(prompt)
        mult_n, clean_text_n = self.token_ex.parse_attn_mult(negative_prompt)
        with autocast(enabled=self.cfgs.dtype == 'amp'):
            emb_n, emb_p = self.te_hook.encode_prompt_to_emb(clean_text_n+clean_text_p).chunk(2)
            emb_p = self.te_hook.mult_attn(emb_p, mult_p)
            emb_n = self.te_hook.mult_attn(emb_n, mult_n)

            to_cpu(self.pipe.text_encoder)
            to_cuda(self.pipe.unet)

            if hasattr(self.pipe.unet, 'input_feeder'):
                for feeder in self.pipe.unet.input_feeder:
                    feeder(ex_input_dict)

            images = self.pipe(prompt_embeds=emb_p, negative_prompt_embeds=emb_n, callback=self.inter_callback, **kwargs).images
        return images

    def inter_callback(self, i, t, latents):
        images = None
        for interface in self.cfgs.interface:
            if interface.show_steps>0 and i%interface.show_steps == 0:
                if self.need_inter_imgs and images is None:
                    images = self.pipe.decode_latents(latents)
                    images = self.pipe.numpy_to_pil(images)
                interface.on_inter_step(i, self.cfgs.infer_args.num_inference_steps, t, latents, images)

    def save_images(self, images, prompt, negative_prompt='', save_cfg=True):
        for interface in self.cfgs.interface:
            interface.on_infer_finish(images, prompt, negative_prompt, self.cfgs_raw if save_cfg else None)

    def vis_to_dir(self, prompt, negative_prompt='', save_cfg=True, **kwargs):
        images = self.vis_images(prompt, negative_prompt, **kwargs)
        self.save_images(images, prompt, negative_prompt, save_cfg=save_cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='')
    args, _ = parser.parse_known_args()
    cfgs = load_config_with_cli(args.cfg, args_list=sys.argv[3:])  # skip --cfg

    if cfgs.seed is not None:
        G = torch.Generator()
        G.manual_seed(cfgs.seed)
    else:
        G = None

    viser = Visualizer(cfgs)
    for i in range(cfgs.num):
        viser.vis_to_dir(prompt=[cfgs.prompt]*cfgs.bs, negative_prompt=[cfgs.neg_prompt]*cfgs.bs,
                         generator=G, save_cfg=cfgs.save.save_cfg, **cfgs.infer_args)
