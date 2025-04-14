import torch
from rainbowneko.infer.workflow import (Actions, PrepareAction, LoopAction, LoadModelAction)
from rainbowneko.ckpt_manager import NekoModelLoader
from rainbowneko.parser import neko_cfg, disable_neko_cfg
from typing import Union, List

from hcpdiff.ckpt_manager import HCPLoraLoader
from hcpdiff.easy import Diffusers_SD, SD15_auto_loader, SDXL_auto_loader
from hcpdiff.workflow import (BuildModelsAction, PrepareDiffusionAction, XformersEnableAction, VaeOptimizeAction, TextHookAction,
                              AttnMultTextEncodeAction, SeedAction, MakeTimestepsAction, MakeLatentAction, DiffusionStepAction,
                              time_iter, DecodeAction, SaveImageAction, LatentResizeAction)

negative_prompt = 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'

## Easy config
@neko_cfg
def build_model(pretrained_model='ckpts/any5', noise_sampler=Diffusers_SD.dpmpp_2m_karras) -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=torch.float16),
        BuildModelsAction(
            model_loader=SD15_auto_loader(
                _partial_=True,
                ckpt_path=pretrained_model,
                noise_sampler=noise_sampler
            )
        ),
    ])

@neko_cfg
def load_parts(info: List[str]) -> Actions:
    acts = []
    for i, path in enumerate(info):
        part_unet = LoadModelAction(cfg={
            f'part_unet_{i}':NekoModelLoader(
                path=path,
                state_prefix='denoiser.'
            )
        }, key_map_in=('denoiser -> model', 'in_preview -> in_preview'))
        part_TE = LoadModelAction(cfg={
            f'part_TE_{i}':NekoModelLoader(
                path=path,
                state_prefix='TE.',
            )
        }, key_map_in=('TE -> model', 'in_preview -> in_preview'))

        with disable_neko_cfg:
            acts.append(part_unet)
            acts.append(part_TE)

    return Actions(acts)

@neko_cfg
def load_lora(info: List[List]) -> Actions:
    lora_acts = []
    for i, item in enumerate(info):
        lora_unet = LoadModelAction(cfg={
            f'lora_unet_{i}':HCPLoraLoader(
                path=item[0],
                state_prefix='denoiser.',
                alpha=item[1],
            )
        }, key_map_in=('denoiser -> model', 'in_preview -> in_preview'))
        lora_TE = LoadModelAction(cfg={
            f'lora_TE_{i}':HCPLoraLoader(
                path=item[0],
                state_prefix='TE.',
                alpha=item[1],
            )
        }, key_map_in=('TE -> model', 'in_preview -> in_preview'))

        with disable_neko_cfg:
            lora_acts.append(lora_unet)
            lora_acts.append(lora_TE)

    return Actions(lora_acts)

@neko_cfg
def optimize_model() -> Actions:
    return Actions([
        PrepareDiffusionAction(),
        XformersEnableAction(),
        VaeOptimizeAction(slicing=True),
    ])

@neko_cfg
def text(prompt, negative_prompt=negative_prompt, bs=4, N_repeats=1, layer_skip=1) -> Actions:
    return Actions([
        TextHookAction(N_repeats=N_repeats, layer_skip=layer_skip),
        AttnMultTextEncodeAction(
            prompt=prompt,
            negative_prompt=negative_prompt,
            bs=bs
        ),
    ])

@neko_cfg
def build_model_SDXL(pretrained_model='ckpts/any5', noise_sampler=Diffusers_SD.dpmpp_2m_karras) -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=torch.float16),
        ## Easy config
        BuildModelsAction(
            model_loader=SDXL_auto_loader(
                _partial_=True,
                ckpt_path=pretrained_model,
                noise_sampler=noise_sampler
            )
        ),
    ])

@neko_cfg
def text_SDXL(prompt, negative_prompt=negative_prompt, bs=4, N_repeats=1, layer_skip=1) -> Actions:
    return Actions([
        TextHookAction(N_repeats=N_repeats, layer_skip=layer_skip, TE_final_norm=False),
        AttnMultTextEncodeAction(
            prompt=prompt,
            negative_prompt=negative_prompt,
            bs=bs
        ),
    ])

@neko_cfg
def config_diffusion(width=512, height=512, seed=None, N_steps=20, strength: float = None) -> Actions:
    return Actions([
        SeedAction(seed),
        MakeTimestepsAction(N_steps=N_steps, strength=strength),
        MakeLatentAction(width=width, height=height)
    ])

@neko_cfg
def diffusion(guidance_scale=7.0) -> Actions:
    return Actions([
        LoopAction(
            iterator=time_iter,
            actions=[
                DiffusionStepAction(guidance_scale=guidance_scale)
            ]
        )
    ])

@neko_cfg
def decode(save_root='output_pipe/') -> Actions:
    return Actions([
        DecodeAction(),
        SaveImageAction(save_root=save_root, image_type='png'),
    ])

@neko_cfg
def resize(width=1024, height=1024):
    return Actions([
        LatentResizeAction(width=width, height=height)
    ])

@neko_cfg
def SD15_t2i(pretrained_model, prompt, negative_prompt=negative_prompt, noise_sampler=Diffusers_SD.dpmpp_2m_karras, bs=4, width=512, height=512,
             seed=None, N_steps=20, guidance_scale=7.0, save_root='output_pipe/', N_repeats=1, layer_skip=1):
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model=pretrained_model, noise_sampler=noise_sampler),
        optimize_model(),
        text(prompt=prompt, negative_prompt=negative_prompt, bs=bs, N_repeats=N_repeats, layer_skip=layer_skip),
        config_diffusion(width=width, height=height, seed=seed, N_steps=N_steps),
        diffusion(guidance_scale=guidance_scale),
        decode(save_root=save_root)
    ]))

@neko_cfg
def SD15_t2i_parts(pretrained_model, parts, prompt, negative_prompt=negative_prompt, noise_sampler=Diffusers_SD.dpmpp_2m_karras, bs=4, width=512, height=512,
             seed=None, N_steps=20, guidance_scale=7.0, save_root='output_pipe/', N_repeats=1, layer_skip=1):
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model=pretrained_model, noise_sampler=noise_sampler),
        load_parts(parts),
        optimize_model(),
        text(prompt=prompt, negative_prompt=negative_prompt, bs=bs, N_repeats=N_repeats, layer_skip=layer_skip),
        config_diffusion(width=width, height=height, seed=seed, N_steps=N_steps),
        diffusion(guidance_scale=guidance_scale),
        decode(save_root=save_root)
    ]))

@neko_cfg
def SD15_t2i_lora(pretrained_model, lora_info, prompt, negative_prompt=negative_prompt, noise_sampler=Diffusers_SD.dpmpp_2m_karras, bs=4,
                  width=512, height=512, seed=None, N_steps=20, guidance_scale=7.0, save_root='output_pipe/', N_repeats=1, layer_skip=1):
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model=pretrained_model, noise_sampler=noise_sampler),
        load_lora(info=lora_info),
        optimize_model(),
        text(prompt=prompt, negative_prompt=negative_prompt, bs=bs, N_repeats=N_repeats, layer_skip=layer_skip),
        config_diffusion(width=width, height=height, seed=seed, N_steps=N_steps),
        diffusion(guidance_scale=guidance_scale),
        decode(save_root=save_root)
    ]))

@neko_cfg
def SDXL_t2i(pretrained_model, prompt, negative_prompt=negative_prompt, noise_sampler=Diffusers_SD.dpmpp_2m_karras, bs=4, width=1024, height=1024,
             seed=None, N_steps=20, guidance_scale=7.0, save_root='output_pipe/', N_repeats=1, layer_skip=1):
    return dict(workflow=Actions(actions=[
        build_model_SDXL(pretrained_model=pretrained_model, noise_sampler=noise_sampler),
        optimize_model(),
        text_SDXL(prompt=prompt, negative_prompt=negative_prompt, bs=bs, N_repeats=N_repeats, layer_skip=layer_skip),
        config_diffusion(width=width, height=height, seed=seed, N_steps=N_steps),
        diffusion(guidance_scale=guidance_scale),
        decode(save_root=save_root)
    ]))

@neko_cfg
def SDXL_t2i_parts(pretrained_model, parts, prompt, negative_prompt=negative_prompt, noise_sampler=Diffusers_SD.dpmpp_2m_karras, bs=4, width=1024, height=1024,
             seed=None, N_steps=20, guidance_scale=7.0, save_root='output_pipe/', N_repeats=1, layer_skip=1):
    return dict(workflow=Actions(actions=[
        build_model_SDXL(pretrained_model=pretrained_model, noise_sampler=noise_sampler),
        load_parts(parts),
        optimize_model(),
        text_SDXL(prompt=prompt, negative_prompt=negative_prompt, bs=bs, N_repeats=N_repeats, layer_skip=layer_skip),
        config_diffusion(width=width, height=height, seed=seed, N_steps=N_steps),
        diffusion(guidance_scale=guidance_scale),
        decode(save_root=save_root)
    ]))


@neko_cfg
def SDXL_t2i_lora(pretrained_model, lora_info, prompt, negative_prompt=negative_prompt, noise_sampler=Diffusers_SD.dpmpp_2m_karras, bs=4,
                  width=1024, height=1024, seed=None, N_steps=20, guidance_scale=7.0, save_root='output_pipe/', N_repeats=1, layer_skip=1):
    return dict(workflow=Actions(actions=[
        build_model_SDXL(pretrained_model=pretrained_model, noise_sampler=noise_sampler),
        load_lora(info=lora_info),
        optimize_model(),
        text_SDXL(prompt=prompt, negative_prompt=negative_prompt, bs=bs, N_repeats=N_repeats, layer_skip=layer_skip),
        config_diffusion(width=width, height=height, seed=seed, N_steps=N_steps),
        diffusion(guidance_scale=guidance_scale),
        decode(save_root=save_root)
    ]))
