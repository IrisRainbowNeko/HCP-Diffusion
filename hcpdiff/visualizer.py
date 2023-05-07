import argparse
import os
import sys

import hydra
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import PIL_INTERPOLATION
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from hcpdiff.models import EmbeddingPTHook, TEEXHook, TokenizerHook
from hcpdiff.utils.cfg_net_tools import load_hcpdiff, make_plugin
from hcpdiff.utils.utils import to_validate_file, load_config_with_cli, load_config
from hcpdiff.utils.img_size_tool import types_support
from torch.cuda.amp import autocast
from PIL import Image
import numpy as np

class UnetHook(): # for controlnet
    def __init__(self, unet):
        self.unet = unet
        self.call_raw = UNet2DConditionModel.__call__
        UNet2DConditionModel.__call__ = self.unet_call

    def unet_call(self, sample, timestep, encoder_hidden_states, **kwargs):
        return self.call_raw(self.unet, sample, timestep, encoder_hidden_states)

class Visualizer:
    def __init__(self, cfgs):
        self.cfgs_raw = cfgs
        self.cfgs = hydra.utils.instantiate(self.cfgs_raw)
        self.cfg_merge = self.cfgs.merge

        pipeline = self.get_pipeline()
        comp = pipeline.from_pretrained(self.cfgs.pretrained_model, safety_checker=None, requires_safety_checker=False).components
        comp.update(self.cfgs.new_components)
        self.pipe = pipeline(**comp)

        if self.cfg_merge:
            self.merge_model()

        self.pipe = self.pipe.to("cuda")
        emb, _ = EmbeddingPTHook.hook_from_dir(self.cfgs.emb_dir, self.pipe.tokenizer, self.pipe.text_encoder, N_repeats=self.cfgs.N_repeats)
        self.te_hook = TEEXHook.hook_pipe(self.pipe, N_repeats=self.cfgs.N_repeats, clip_skip=self.cfgs.clip_skip)
        self.token_ex = TokenizerHook(self.pipe.tokenizer)
        UnetHook(self.pipe.unet)

        if is_xformers_available():
            self.pipe.unet.enable_xformers_memory_efficient_attention()
            # self.te_hook.enable_xformers()

    def get_pipeline(self):
        if self.cfgs.condition is None:
            return StableDiffusionPipeline
        else:
            if self.cfgs.condition.type=='i2i':
                return StableDiffusionImg2ImgPipeline
            elif self.cfgs.condition.type=='controlnet':
                return StableDiffusionPipeline
            else:
                raise NotImplementedError(f'No condition type named {self.cfgs.condition.type}')

    def merge_model(self):
        if 'plugin_cfg' in self.cfg_merge: # Build plugins
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
                image = np.array(image).astype(np.float32) / 255.0
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

        mult_p, clean_text_p = self.token_ex.parse_attn_mult(prompt)
        mult_n, clean_text_n = self.token_ex.parse_attn_mult(negative_prompt)
        with autocast(enabled=self.cfgs.fp16):
            emb_n, emb_p = self.te_hook.encode_prompt_to_emb(clean_text_n+clean_text_p).chunk(2)
            emb_p = self.te_hook.mult_attn(emb_p, mult_p)
            emb_n = self.te_hook.mult_attn(emb_n, mult_n)

            if hasattr(self.pipe.unet, 'input_feeder'):
                for feeder in self.pipe.unet.input_feeder:
                    feeder(ex_input_dict)

            images = self.pipe(prompt_embeds=emb_p, negative_prompt_embeds=emb_n, **kwargs).images
        return images

    def save_images(self, images, root, prompt, negative_prompt='', save_cfg=True):
        os.makedirs(root, exist_ok=True)
        num_img_exist = max([int(x.split('-', 1)[0]) for x in os.listdir(root) if x.rsplit('.', 1)[-1] in types_support]) + 1

        for p, pn, img in zip(prompt, negative_prompt, images):
            img.save(os.path.join(root, f"{num_img_exist}-{to_validate_file(prompt[0])}.{self.cfgs.save.image_type}"), quality=self.cfgs.save.quality)

            if save_cfg:
                with open(os.path.join(root, f"{num_img_exist}-info.yaml"), 'w', encoding='utf-8') as f:
                    f.write(OmegaConf.to_yaml(self.cfgs_raw))
            num_img_exist += 1

    def vis_to_dir(self, root, prompt, negative_prompt='', save_cfg=True, **kwargs):
        images = self.vis_images(prompt, negative_prompt, **kwargs)
        self.save_images(images, root, prompt, negative_prompt, save_cfg=save_cfg)

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
