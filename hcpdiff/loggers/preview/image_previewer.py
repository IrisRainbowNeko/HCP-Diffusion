import os
from contextlib import contextmanager
from typing import List

import hydra
import torch
from diffusers import PNDMScheduler
from torch.cuda.amp import autocast

from hcpdiff.visualizer import Visualizer
from hcpdiff.models import TokenizerHook
from hcpdiff.utils.utils import prepare_seed, load_config

class ImagePreviewer(Visualizer):
    def __init__(self, infer_cfg, exp_dir, te_hook,
                 unet, TE, tokenizer, vae, save_cfg=False, preview_dir='preview'):
        self.exp_dir = exp_dir
        self.cfgs_raw = load_config(infer_cfg)
        self.cfgs = hydra.utils.instantiate(self.cfgs_raw)
        self.save_cfg = save_cfg

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

        self.preview_dir = preview_dir
        os.makedirs(os.path.join(exp_dir, preview_dir), exist_ok=True)

    @contextmanager
    def infer_optimize(self):
        if getattr(self.cfgs, 'vae_optimize', None) is not None:
            if self.cfgs.vae_optimize.tiling:
                self.pipe.vae.enable_tiling()
            if self.cfgs.vae_optimize.slicing:
                self.pipe.vae.enable_slicing()
        vae_device = self.pipe.vae.device
        self.pipe.vae.to(self.pipe.unet.device)
        yield
        self.pipe.vae.to(vae_device)
        self.pipe.vae.disable_tiling()
        self.pipe.vae.disable_slicing()

    def preview(self):
        prompt_all, negative_prompt_all, seeds_all, images_all = [], [], [], []
        with self.infer_optimize():
            for i in range(self.cfgs.num):
                prompt = self.cfgs.prompt[i*self.cfgs.bs:(i+1)*self.cfgs.bs] if isinstance(self.cfgs.prompt, list) \
                    else [self.cfgs.prompt]*self.cfgs.bs
                negative_prompt = self.cfgs.neg_prompt[i*self.cfgs.bs:(i+1)*self.cfgs.bs] if isinstance(self.cfgs.neg_prompt, list) \
                    else [self.cfgs.neg_prompt]*self.cfgs.bs
                seeds = self.seeds[i*self.cfgs.bs:(i+1)*self.cfgs.bs]
                images = self.vis_images(prompt=prompt, negative_prompt=negative_prompt, seeds=seeds,
                                         **self.cfgs.infer_args)

                prompt_all += prompt
                negative_prompt_all += negative_prompt
                seeds_all += seeds
                images_all += images

        return prompt_all, negative_prompt_all, seeds_all, images_all, self.cfgs_raw if self.save_cfg else None

    @torch.no_grad()
    def vis_images(self, prompt, negative_prompt='', seeds: List[int] = None, **kwargs):
        G = prepare_seed(seeds or [None]*len(prompt))

        ex_input_dict, pipe_input_dict = self.get_ex_input()
        kwargs.update(pipe_input_dict)

        mult_p, clean_text_p = self.token_ex.parse_attn_mult(prompt)
        mult_n, clean_text_n = self.token_ex.parse_attn_mult(negative_prompt)
        with autocast(enabled=self.cfgs.dtype == 'amp'):
            emb, pooled_output = self.te_hook.encode_prompt_to_emb(clean_text_n+clean_text_p)
            emb_n, emb_p = emb.chunk(2)
            emb_p = self.te_hook.mult_attn(emb_p, mult_p)
            emb_n = self.te_hook.mult_attn(emb_n, mult_n)

            if hasattr(self.pipe.unet, 'input_feeder'):
                for feeder in self.pipe.unet.input_feeder:
                    feeder(ex_input_dict)

            images = self.pipe(prompt_embeds=emb_p, negative_prompt_embeds=emb_n, callback=self.inter_callback, generator=G,
                               pooled_output=pooled_output[-1], **kwargs).images
        return images
