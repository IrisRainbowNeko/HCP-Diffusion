import argparse
import os
from typing import List

import hydra
import torch
from sfast.compilers.diffusion_pipeline_compiler import (compile_unet, CompilationConfig)
from torch.cuda.amp import autocast

from hcpdiff import Visualizer
from hcpdiff.models import TokenizerHook
from hcpdiff.models.compose import ComposeTEEXHook, ComposeEmbPTHook, ComposeTextEncoder
from hcpdiff.utils.net_utils import to_cuda
from hcpdiff.utils.utils import load_config_with_cli, prepare_seed, is_list, pad_attn_bias

class VisualizerFast(Visualizer):
    dtype_dict = {'fp32':torch.float32, 'fp16':torch.float16, 'bf16':torch.bfloat16}

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

        # self.pipe = self.pipe.to(torch_dtype=self.dtype)

        if isinstance(self.pipe.text_encoder, ComposeTextEncoder):
            self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)

        if 'save_model' in self.cfgs and self.cfgs.save_model is not None:
            self.save_model(self.cfgs.save_model)
            os._exit(0)

        self.build_optimize()
        self.compile_model()

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
                                                 clip_final_norm=self.cfgs.clip_final_norm, use_attention_mask=self.cfgs.encoder_attention_mask)
        self.token_ex = TokenizerHook(self.pipe.tokenizer)

    def compile_model(self):
        # compile model
        config = CompilationConfig.Default()
        config.enable_xformers = False
        try:
            import xformers
            config.enable_xformers = True
        except ImportError:
            print('xformers not installed, skip')
        # NOTE:
        # When GPU VRAM is insufficient or the architecture is too old, Triton might be slow.
        # Disable Triton if you encounter this problem.
        try:
            import tritonx
            config.enable_triton = True
        except ImportError:
            print('Triton not installed, skip')
        config.enable_cuda_graph = True

        self.pipe.unet = compile_unet(self.pipe.unet, config)

    @torch.inference_mode()
    def vis_images(self, prompt, negative_prompt='', seeds: List[int] = None, **kwargs):
        G = prepare_seed(seeds or [None]*len(prompt))

        ex_input_dict, pipe_input_dict = self.get_ex_input()
        kwargs.update(pipe_input_dict)

        to_cuda(self.pipe.text_encoder)

        mult_p, clean_text_p = self.token_ex.parse_attn_mult(prompt)
        mult_n, clean_text_n = self.token_ex.parse_attn_mult(negative_prompt)
        with autocast(enabled=self.cfgs.amp, dtype=self.dtype):
            emb, pooled_output, attention_mask = self.te_hook.encode_prompt_to_emb(clean_text_n+clean_text_p)
            if self.cfgs.encoder_attention_mask:
                emb, attention_mask = pad_attn_bias(emb, attention_mask)
            else:
                attention_mask = None
            emb_n, emb_p = emb.chunk(2)
            emb_p = self.te_hook.mult_attn(emb_p, mult_p)
            emb_n = self.te_hook.mult_attn(emb_n, mult_n)

            # to_cpu(self.pipe.text_encoder)
            # to_cuda(self.pipe.unet)

            if hasattr(self.pipe.unet, 'input_feeder'):
                for feeder in self.pipe.unet.input_feeder:
                    feeder(ex_input_dict)

            images = self.pipe(prompt_embeds=emb_p, negative_prompt_embeds=emb_n, callback=None, generator=G,
                               pooled_output=pooled_output[-1], encoder_attention_mask=attention_mask, **kwargs).images
        return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast HCP Diffusion Inference')
    parser.add_argument('--cfg', type=str, default='cfgs/infer/text2img.yaml')
    args, cfg_args = parser.parse_known_args()
    cfgs = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg

    if cfgs.seed is not None:
        if is_list(cfgs.seed):
            assert len(cfgs.seed) == cfgs.num*cfgs.bs, 'seed list length should be equal to num*bs'
            seeds = list(cfgs.seed)
        else:
            seeds = list(range(cfgs.seed, cfgs.seed+cfgs.num*cfgs.bs))
    else:
        seeds = [None]*(cfgs.num*cfgs.bs)

    viser = VisualizerFast(cfgs)

    for i in range(cfgs.num):
        prompt = cfgs.prompt[i*cfgs.bs:(i+1)*cfgs.bs] if is_list(cfgs.prompt) else [cfgs.prompt]*cfgs.bs
        negative_prompt = cfgs.neg_prompt[i*cfgs.bs:(i+1)*cfgs.bs] if is_list(cfgs.neg_prompt) else [cfgs.neg_prompt]*cfgs.bs
        viser.vis_to_dir(prompt=prompt, negative_prompt=negative_prompt,
                         seeds=seeds[i*cfgs.bs:(i+1)*cfgs.bs], save_cfg=cfgs.save.save_cfg, **cfgs.infer_args)
