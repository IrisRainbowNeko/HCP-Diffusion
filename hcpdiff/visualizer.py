import argparse
import os
import sys

import hydra
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from hcpdiff.models import EmbeddingPTHook, TEEXHook, TokenizerHook
from hcpdiff.utils.cfg_net_tools import load_hcpdiff
from hcpdiff.utils.utils import to_validate_file, load_config_with_cli
from hcpdiff.utils.img_size_tool import types_support
from torch.cuda.amp import autocast

class Visualizer:
    def __init__(self, cfgs):
        self.cfgs_raw = cfgs
        self.cfgs = hydra.utils.instantiate(self.cfgs_raw)
        self.cfg_merge = cfgs.merge

        comp = StableDiffusionPipeline.from_pretrained(cfgs.pretrained_model, safety_checker=None, requires_safety_checker=False).components
        comp.update(cfgs.new_components)
        self.pipe = StableDiffusionPipeline(**comp)

        if self.cfg_merge:
            self.merge_model()

        self.pipe = self.pipe.to("cuda")
        emb, _ = EmbeddingPTHook.hook_from_dir(cfgs.emb_dir, self.pipe.tokenizer, self.pipe.text_encoder, N_repeats=cfgs.N_repeats)
        self.te_hook = TEEXHook.hook_pipe(self.pipe, N_repeats=cfgs.N_repeats, clip_skip=cfgs.clip_skip)
        self.token_ex = TokenizerHook(self.pipe.tokenizer)

        if is_xformers_available():
            self.pipe.unet.enable_xformers_memory_efficient_attention()
            # self.te_hook.enable_xformers()

    def merge_model(self):
        for cfg_group in self.cfg_merge.values():
            if hasattr(cfg_group, 'type'):
                if cfg_group.type == 'unet':
                    load_hcpdiff(self.pipe.unet, cfg_group)
                elif cfg_group.type == 'TE':
                    load_hcpdiff(self.pipe.text_encoder, cfg_group)

    def set_scheduler(self, scheduler):
        self.pipe.scheduler = scheduler

    @torch.no_grad()
    def vis_to_dir(self, root, prompt, negative_prompt='', save_cfg=True, **kwargs):
        os.makedirs(root, exist_ok=True)
        num_img_exist = len([x for x in os.listdir(root) if x.rsplit('.', 1)[-1] in types_support])

        mult_p, clean_text_p = self.token_ex.parse_attn_mult(prompt)
        mult_n, clean_text_n = self.token_ex.parse_attn_mult(negative_prompt)
        with autocast(enabled=self.cfgs.fp16):
            emb_n, emb_p = self.te_hook.encode_prompt_to_emb(clean_text_n + clean_text_p).chunk(2)
            emb_p = self.te_hook.mult_attn(emb_p, mult_p)
            emb_n = self.te_hook.mult_attn(emb_n, mult_n)
            images = self.pipe(prompt_embeds=emb_p, negative_prompt_embeds=emb_n, **kwargs).images

        for p, pn, img in zip(prompt, negative_prompt, images):
            img.save(os.path.join(root, f"{num_img_exist}-{to_validate_file(prompt[0])}.{self.cfgs.save.image_type}"), quality=self.cfgs.save.quality)

            if save_cfg:
                with open(os.path.join(root, f"{num_img_exist}-info.yaml"), 'w', encoding='utf-8') as f:
                    f.write(OmegaConf.to_yaml(self.cfgs_raw))
            num_img_exist += 1

    def show_latent(self, prompt, negative_prompt='', **kwargs):
        emb_n, emb_p = self.te_hook.encode_prompt_to_emb(negative_prompt + prompt).chunk(2)
        emb_p = self.te_hook.mult_attn(emb_p, self.token_ex.parse_attn_mult(prompt))
        emb_n = self.te_hook.mult_attn(emb_n, self.token_ex.parse_attn_mult(negative_prompt))
        images = self.pipe(prompt_embeds=emb_p, negative_prompt_embeds=emb_n, output_type='latent', **kwargs).images

        for img in images:
            plt.figure()
            for i, feat in enumerate(img):
                plt.subplot(221 + i)
                plt.imshow(feat)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='')
    args, _ = parser.parse_known_args()
    cfgs = load_config_with_cli(args.cfg, args_list=sys.argv[3:])  # skip --cfg

    os.makedirs(cfgs.out_dir, exist_ok=True)
    if cfgs.seed is not None:
        G = torch.Generator()
        G.manual_seed(cfgs.seed)
    else:
        G = None

    viser = Visualizer(cfgs)
    for i in range(cfgs.num):
        viser.vis_to_dir(cfgs.out_dir, prompt=[cfgs.prompt] * cfgs.bs, negative_prompt=[cfgs.neg_prompt] * cfgs.bs,
                         generator=G, save_cfg=cfgs.save.save_cfg, **cfgs.infer_args)
