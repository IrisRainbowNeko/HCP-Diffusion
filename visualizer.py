import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
import argparse
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

import os
from utils.utils import to_validate_file, load_config, str2bool
from utils.cfg_net_tools import load_hcpdiff
from models import EmbeddingPTHook, TEEXHook, TokenizerHook

class Visualizer:
    def __init__(self, pretrained, new_components={}, emb_dir='embs/', N_repeats=3, cfg_merge=None):
        self.cfg_merge=cfg_merge

        comp = StableDiffusionPipeline.from_pretrained(pretrained, safety_checker=None, requires_safety_checker=False).components
        comp.update(new_components)
        self.pipe = StableDiffusionPipeline(**comp)

        if cfg_merge:
            self.merge_model()

        self.pipe = self.pipe.to("cuda")
        emb, _ = EmbeddingPTHook.hook_from_dir(emb_dir, self.pipe.tokenizer, self.pipe.text_encoder, N_repeats=N_repeats)
        self.te_hook = TEEXHook.hook_pipe(self.pipe, N_repeats=N_repeats)
        self.token_ex = TokenizerHook(self.pipe.tokenizer)

        if is_xformers_available():
            self.pipe.unet.enable_xformers_memory_efficient_attention()
            #self.te_hook.enable_xformers()

    def merge_model(self):
        cfgs = load_config(self.cfg_merge)
        self.cfgs = cfgs
        for cfg_group in cfgs.values():
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
        num_img_exist = len([x for x in os.listdir(root) if x.endswith('.png')])

        emb_n, emb_p = self.te_hook.encode_prompt_to_emb(negative_prompt+prompt).chunk(2)
        emb_p = self.te_hook.mult_attn(emb_p, self.token_ex.parse_attn_mult(prompt))
        emb_n = self.te_hook.mult_attn(emb_n, self.token_ex.parse_attn_mult(negative_prompt))
        images = self.pipe(prompt_embeds=emb_p, negative_prompt_embeds=emb_n, **kwargs).images

        for p, pn, img in zip(prompt, negative_prompt, images):
            img.save(os.path.join(root, f"{num_img_exist}-{to_validate_file(prompt[0])}.png"))

            if save_cfg:
                info = OmegaConf.create()
                info.prompt = p
                info.neg_prompt = pn
                info = OmegaConf.merge(info, self.cfgs)
                with open(os.path.join(root, f"{num_img_exist}-info.yaml"), 'w', encoding='utf-8') as f:
                    f.write(OmegaConf.to_yaml(info))
            num_img_exist+=1

    def show_latent(self, prompt, negative_prompt='', **kwargs):
        emb_n, emb_p = self.te_hook.encode_prompt_to_emb(negative_prompt+prompt).chunk(2)
        emb_p = self.te_hook.mult_attn(emb_p, self.token_ex.parse_attn_mult(prompt))
        emb_n = self.te_hook.mult_attn(emb_n, self.token_ex.parse_attn_mult(negative_prompt))
        images = self.pipe(prompt_embeds=emb_p, negative_prompt_embeds=emb_n, output_type='latent', **kwargs).images

        for img in images:
            plt.figure()
            for i, feat in enumerate(img):
                plt.subplot(221+i)
                plt.imshow(feat)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--neg_prompt', type=str, default='lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry')
    parser.add_argument('--out_dir', type=str, default='output/')
    parser.add_argument('--emb_dir', type=str, default='embs/')
    parser.add_argument('--cfg_merge', type=str, default=None)
    parser.add_argument('--N_repeat', type=int, default=3)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_cfg', type=str2bool, default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.seed is not None:
        G=torch.Generator()
        G.manual_seed(args.seed)
    else:
        G=None

    viser = Visualizer(args.pretrained_model, emb_dir=args.emb_dir, N_repeats=args.N_repeat, cfg_merge=args.cfg_merge)
    for i in range(args.num):
        viser.vis_to_dir(args.out_dir, prompt=[args.prompt]*args.bs, negative_prompt=[args.neg_prompt]*args.bs, width=args.W, height=args.H,
                     generator=G, save_cfg=args.save_cfg)