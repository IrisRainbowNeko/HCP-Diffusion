import argparse
import os
import random
import sys
from typing import List

import hydra
import torch
from PIL import Image
from accelerate import infer_auto_device_map, dispatch_model
from diffusers.utils.import_utils import is_xformers_available
from torch.cuda.amp import autocast

from hcpdiff.models import EmbeddingPTHook, TEEXHook, TokenizerHook, LoraBlock
from hcpdiff.models.compose import ComposeTEEXHook, ComposeEmbPTHook, ComposeTextEncoder
from hcpdiff.utils.cfg_net_tools import load_hcpdiff, make_plugin
from hcpdiff.utils.net_utils import to_cpu, to_cuda, auto_tokenizer, auto_text_encoder
from hcpdiff.utils.pipe_hook import HookPipe_T2I, HookPipe_I2I, HookPipe_Inpaint
from hcpdiff.utils.utils import load_config_with_cli, load_config, size_to_int, int_to_size, prepare_seed

class Visualizer:
    dtype_dict = {'fp32':torch.float32, 'amp':torch.float32, 'fp16':torch.float16, 'bf16':torch.bfloat16}

    def __init__(self, cfgs):
        self.cfgs_raw = cfgs
        self.cfgs = hydra.utils.instantiate(self.cfgs_raw)
        self.cfg_merge = self.cfgs.merge
        self.offload = 'offload' in self.cfgs and self.cfgs.offload is not None
        self.dtype = self.dtype_dict[self.cfgs.dtype]

        self.need_inter_imgs = any(item.need_inter_imgs for item in self.cfgs.interface)

        self.pipe = self.load_model(self.cfgs.pretrained_model)

        if self.cfg_merge:
            self.merge_model()

        self.pipe = self.pipe.to(torch_dtype=self.dtype)

        if isinstance(self.pipe.text_encoder, ComposeTextEncoder):
            self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)

        if 'save_model' in self.cfgs and self.cfgs.save_model is not None:
            self.save_model(self.cfgs.save_model)
            os._exit(0)

        self.build_optimize()

    def load_model(self, pretrained_model):
        pipeline = self.get_pipeline()
        te = auto_text_encoder(pretrained_model).from_pretrained(pretrained_model, subfolder="text_encoder", torch_dtype=self.dtype)
        tokenizer = auto_tokenizer(pretrained_model).from_pretrained(pretrained_model, subfolder="tokenizer", use_fast=False)

        return pipeline.from_pretrained(pretrained_model, safety_checker=None, requires_safety_checker=False,
                                        text_encoder=te, tokenizer=tokenizer,
                                        torch_dtype=self.dtype, **self.cfgs.new_components)

    def build_optimize(self):
        if self.offload:
            self.build_offload(self.cfgs.offload)
        else:
            self.pipe.unet.to('cuda')
        self.build_vae_offload()

        if getattr(self.cfgs, 'vae_optimize', None) is not None:
            if self.cfgs.vae_optimize.tiling:
                self.pipe.vae.enable_tiling()
            if self.cfgs.vae_optimize.slicing:
                self.pipe.vae.enable_slicing()

        self.emb_hook, _ = ComposeEmbPTHook.hook_from_dir(self.cfgs.emb_dir, self.pipe.tokenizer, self.pipe.text_encoder,
                                                         N_repeats=self.cfgs.N_repeats)
        self.te_hook = ComposeTEEXHook.hook_pipe(self.pipe, N_repeats=self.cfgs.N_repeats, clip_skip=self.cfgs.clip_skip,
                                                 clip_final_norm=self.cfgs.clip_final_norm)
        self.token_ex = TokenizerHook(self.pipe.tokenizer)

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
            elif self.cfgs.condition.type == 'inpaint':
                pipe_cls = HookPipe_Inpaint
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

    def build_vae_offload(self):
        def vae_decode_offload(latents, return_dict=True, decode_raw=self.pipe.vae.decode):
            if self.need_inter_imgs:
                to_cuda(self.pipe.vae)
                res = decode_raw(latents, return_dict=return_dict)
            else:
                to_cpu(self.pipe.unet)

                if self.offload and self.cfgs.offload.vae_cpu:
                    self.pipe.vae.to(dtype=torch.float32)
                    res = decode_raw(latents.cpu().to(dtype=torch.float32), return_dict=return_dict)
                else:
                    to_cuda(self.pipe.vae)
                    res = decode_raw(latents.to(dtype=torch.float32), return_dict=return_dict)

                to_cpu(self.pipe.vae)
                to_cuda(self.pipe.unet)
            return res

        self.pipe.vae.decode = vae_decode_offload

        def vae_encode_offload(x, return_dict=True, encode_raw=self.pipe.vae.encode):
            to_cuda(self.pipe.vae)
            res = encode_raw(x, return_dict=return_dict)
            to_cpu(self.pipe.vae)
            return res

        self.pipe.vae.encode = vae_encode_offload

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

    def get_ex_input(self):
        ex_input_dict, pipe_input_dict = {}, {}
        if self.cfgs.condition is not None:
            if self.cfgs.condition.type == 'i2i':
                pipe_input_dict['image'] = Image.open(self.cfgs.condition.image).convert('RGB')
            elif self.cfgs.condition.type == 'inpaint':
                pipe_input_dict['image'] = Image.open(self.cfgs.condition.image).convert('RGB')
                pipe_input_dict['mask_image'] = Image.open(self.cfgs.condition.mask).convert('L')

        if getattr(self.cfgs, 'ex_input', None) is not None:
            for key, processor in self.cfgs.ex_input.items():
                ex_input_dict[key] = processor(self.cfgs.infer_args.width, self.cfgs.infer_args.height, self.cfgs.bs*2, 'cuda', self.dtype)
        return ex_input_dict, pipe_input_dict

    @torch.no_grad()
    def vis_images(self, prompt, negative_prompt='', seeds: List[int] = None, **kwargs):
        G = prepare_seed(seeds or [None]*len(prompt))

        ex_input_dict, pipe_input_dict = self.get_ex_input()
        kwargs.update(pipe_input_dict)

        to_cuda(self.pipe.text_encoder)

        mult_p, clean_text_p = self.token_ex.parse_attn_mult(prompt)
        mult_n, clean_text_n = self.token_ex.parse_attn_mult(negative_prompt)
        with autocast(enabled=self.cfgs.dtype == 'amp'):
            emb, pooled_output = self.te_hook.encode_prompt_to_emb(clean_text_n+clean_text_p)
            emb_n, emb_p = emb.chunk(2)
            emb_p = self.te_hook.mult_attn(emb_p, mult_p)
            emb_n = self.te_hook.mult_attn(emb_n, mult_n)

            to_cpu(self.pipe.text_encoder)
            to_cuda(self.pipe.unet)

            if hasattr(self.pipe.unet, 'input_feeder'):
                for feeder in self.pipe.unet.input_feeder:
                    feeder(ex_input_dict)

            images = self.pipe(prompt_embeds=emb_p, negative_prompt_embeds=emb_n, callback=self.inter_callback, generator=G,
                               pooled_output=pooled_output[-1], **kwargs).images
        return images

    def inter_callback(self, i, t, num_t, latents):
        images = None
        interrupt = False
        for interface in self.cfgs.interface:
            if interface.show_steps>0 and i%interface.show_steps == 0:
                if self.need_inter_imgs and images is None:
                    images = self.pipe.decode_latents(latents)
                    images = self.pipe.numpy_to_pil(images)
                feed_back = interface.on_inter_step(i, num_t, t, latents, images)
                interrupt |= bool(feed_back)
        return interrupt

    def save_images(self, images, prompt, negative_prompt='', seeds: List[int] = None):
        for interface in self.cfgs.interface:
            interface.on_infer_finish(images, prompt, negative_prompt, self.cfgs_raw, seeds=seeds)

    def vis_to_dir(self, prompt, negative_prompt='', seeds: List[int] = None, **kwargs):
        seeds = [s or random.randint(0, 1 << 30) for s in seeds]

        images = self.vis_images(prompt, negative_prompt, seeds=seeds, **kwargs)
        self.save_images(images, prompt, negative_prompt, seeds=seeds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='')
    args, cfg_args = parser.parse_known_args()
    cfgs = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg

    if cfgs.seed is not None:
        seeds = list(range(cfgs.seed, cfgs.seed+cfgs.num*cfgs.bs))
    else:
        seeds = [None]*(cfgs.num*cfgs.bs)

    viser = Visualizer(cfgs)
    for i in range(cfgs.num):
        viser.vis_to_dir(prompt=[cfgs.prompt]*cfgs.bs, negative_prompt=[cfgs.neg_prompt]*cfgs.bs,
                         seeds=seeds[i*cfgs.bs:(i+1)*cfgs.bs], save_cfg=cfgs.save.save_cfg, **cfgs.infer_args)
