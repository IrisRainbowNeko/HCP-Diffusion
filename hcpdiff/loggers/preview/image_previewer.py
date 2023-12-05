from contextlib import contextmanager
from typing import List

import hydra
import torch
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.hooks import remove_hook_from_module
from diffusers import PNDMScheduler
from torch.cuda.amp import autocast

from hcpdiff.models import TokenizerHook
from hcpdiff.utils.net_utils import to_cpu
from hcpdiff.utils.utils import prepare_seed, load_config, size_to_int, int_to_size
from hcpdiff.utils.utils import to_validate_file
from hcpdiff.visualizer import Visualizer

class ImagePreviewer(Visualizer):
    def __init__(self, infer_cfg, exp_dir, te_hook,
                 unet, TE, tokenizer, vae, save_cfg=False):
        self.exp_dir = exp_dir
        self.cfgs_raw = load_config(infer_cfg)
        self.cfgs = hydra.utils.instantiate(self.cfgs_raw)
        self.save_cfg = save_cfg
        self.offload = 'offload' in self.cfgs and self.cfgs.offload is not None
        self.dtype = self.dtype_dict[self.cfgs.dtype]

        if getattr(self.cfgs.new_components, 'scheduler', None) is None:
            scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')
        else:
            scheduler = self.cfgs.new_components.scheduler

        pipe_cls = self.get_pipeline()
        self.pipe = pipe_cls(vae=vae, text_encoder=TE, tokenizer=tokenizer, unet=unet, scheduler=scheduler, feature_extractor=None,
                             safety_checker=None, requires_safety_checker=False)

        self.token_ex = TokenizerHook(tokenizer)
        self.te_hook = te_hook

        if self.cfgs.seed is not None:
            self.seeds = list(range(self.cfgs.seed, self.cfgs.seed+self.cfgs.num*self.cfgs.bs))
        else:
            self.seeds = [None]*(self.cfgs.num*self.cfgs.bs)

    def build_vae_offload(self, offload_cfg):
        vram = size_to_int(offload_cfg.max_VRAM)
        if not offload_cfg.vae_cpu:
            device_map = infer_auto_device_map(self.pipe.vae, max_memory={0:int_to_size(vram >> 5), "cpu":offload_cfg.max_RAM}, dtype=torch.float32)
            self.pipe.vae = dispatch_model(self.pipe.vae, device_map)
        else:
            to_cpu(self.pipe.vae)
            self.vae_decode_raw = self.pipe.vae.decode

            def vae_decode_offload(latents, return_dict=True, decode_raw=self.pipe.vae.decode):
                self.pipe.vae.to(dtype=torch.float32)
                res = decode_raw(latents.cpu().to(dtype=torch.float32), return_dict=return_dict)
                return res

            self.pipe.vae.decode = vae_decode_offload

            self.vae_encode_raw = self.pipe.vae.encode

            def vae_encode_offload(x, return_dict=True, encode_raw=self.pipe.vae.encode):
                self.pipe.vae.to(dtype=torch.float32)
                res = encode_raw(x.cpu().to(dtype=torch.float32), return_dict=return_dict)
                return res

            self.pipe.vae.encode = vae_encode_offload

    def remove_vae_offload(self, offload_cfg):
        if not offload_cfg.vae_cpu:
            remove_hook_from_module(self.pipe.vae, recurse=True)
        else:
            self.pipe.vae.encode = self.vae_encode_raw
            self.pipe.vae.decode = self.vae_decode_raw

    @contextmanager
    def infer_optimize(self):
        if getattr(self.cfgs, 'vae_optimize', None) is not None:
            if self.cfgs.vae_optimize.tiling:
                self.pipe.vae.enable_tiling()
            if self.cfgs.vae_optimize.slicing:
                self.pipe.vae.enable_slicing()
        vae_device = self.pipe.vae.device
        if self.offload:
            self.build_vae_offload(self.cfgs.offload)
        else:
            self.pipe.vae.to(self.pipe.unet.device)

        yield

        if self.offload:
            self.remove_vae_offload(self.cfgs.offload)
        self.pipe.vae.to(vae_device)
        self.pipe.vae.disable_tiling()
        self.pipe.vae.disable_slicing()

    def preview(self):
        image_list, info_list = [], []
        with self.infer_optimize():
            for i in range(self.cfgs.num):
                prompt = self.cfgs.prompt[i*self.cfgs.bs:(i+1)*self.cfgs.bs] if isinstance(self.cfgs.prompt, list) \
                    else [self.cfgs.prompt]*self.cfgs.bs
                negative_prompt = self.cfgs.neg_prompt[i*self.cfgs.bs:(i+1)*self.cfgs.bs] if isinstance(self.cfgs.neg_prompt, list) \
                    else [self.cfgs.neg_prompt]*self.cfgs.bs
                seeds = self.seeds[i*self.cfgs.bs:(i+1)*self.cfgs.bs]
                images = self.vis_images(prompt=prompt, negative_prompt=negative_prompt, seeds=seeds,
                                         **self.cfgs.infer_args)
                for prompt_i, negative_prompt_i, seed in zip(prompt, negative_prompt, seeds):
                    info_list.append({
                        'prompt':prompt_i,
                        'negative_prompt':negative_prompt_i,
                        'seed':seed,
                    })
                image_list += images

        return image_list, info_list

    def preview_dict(self):
        image_list, info_list = self.preview()
        imgs = {f'{info["seed"]}-{to_validate_file(info["prompt"])}':img for img, info in zip(image_list, info_list)}
        return imgs

    @torch.no_grad()
    def vis_images(self, prompt, negative_prompt='', seeds: List[int] = None, **kwargs):
        G = prepare_seed(seeds or [None]*len(prompt))

        ex_input_dict, pipe_input_dict = self.get_ex_input()
        kwargs.update(pipe_input_dict)

        mult_p, clean_text_p = self.token_ex.parse_attn_mult(prompt)
        mult_n, clean_text_n = self.token_ex.parse_attn_mult(negative_prompt)
        with autocast(enabled=self.cfgs.amp, dtype=self.dtype):
            emb, pooled_output, attention_mask = self.te_hook.encode_prompt_to_emb(clean_text_n+clean_text_p)
            if not self.cfgs.encoder_attention_mask:
                attention_mask = None
            emb_n, emb_p = emb.chunk(2)
            emb_p = self.te_hook.mult_attn(emb_p, mult_p)
            emb_n = self.te_hook.mult_attn(emb_n, mult_n)

            if hasattr(self.pipe.unet, 'input_feeder'):
                for feeder in self.pipe.unet.input_feeder:
                    feeder(ex_input_dict)

            if pooled_output is not None:
                pooled_output = pooled_output[-1]

            images = self.pipe(prompt_embeds=emb_p, negative_prompt_embeds=emb_n, callback=self.inter_callback, generator=G,
                               pooled_output=pooled_output, encoder_attention_mask=attention_mask, **kwargs).images
        return images
