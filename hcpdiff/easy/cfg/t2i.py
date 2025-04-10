import torch
from rainbowneko.infer.workflow import (Actions, PrepareAction, LoopAction)
from rainbowneko.parser import neko_cfg

from hcpdiff.easy import Diffusers_SD, SD15_auto_loader, SDXL_auto_loader
from hcpdiff.workflow import (BuildModelsAction, PrepareDiffusionAction, XformersEnableAction, VaeOptimizeAction, TextHookAction,
                              AttnMultTextEncodeAction, SeedAction, MakeTimestepsAction, MakeLatentAction, DiffusionStepAction,
                              time_iter,
                              DecodeAction, SaveImageAction)

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
def optimize_model() -> Actions:
    return Actions([
        PrepareDiffusionAction(),
        XformersEnableAction(),
        VaeOptimizeAction(slicing=True),
    ])

@neko_cfg
def text(prompt, negative_prompt=negative_prompt, bs=4) -> Actions:
    return Actions([
        TextHookAction(N_repeats=1, layer_skip=1),
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
def text_SDXL(prompt, negative_prompt=negative_prompt, bs=4) -> Actions:
    return Actions([
        TextHookAction(N_repeats=1, layer_skip=1, TE_final_norm=False),
        AttnMultTextEncodeAction(
            prompt=prompt,
            negative_prompt=negative_prompt,
            bs=bs
        ),
    ])

@neko_cfg
def config_diffusion(width=512, height=512, seed=42, N_steps=20) -> Actions:
    return Actions([
        SeedAction(seed),
        MakeTimestepsAction(N_steps=N_steps),
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
def SD15_t2i(pretrained_model, prompt, negative_prompt=negative_prompt, noise_sampler=Diffusers_SD.dpmpp_2m_karras, bs=4, width=512, height=512,
             seed=42, N_steps=20, guidance_scale=7.0, save_root='output_pipe/'):
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model=pretrained_model, noise_sampler=noise_sampler),
        optimize_model(),
        text(prompt=prompt, negative_prompt=negative_prompt, bs=bs),
        config_diffusion(width=width, height=height, seed=seed, N_steps=N_steps),
        diffusion(guidance_scale=guidance_scale),
        decode(save_root=save_root)
    ]))


@neko_cfg
def SDXL_t2i(pretrained_model, prompt, negative_prompt=negative_prompt, noise_sampler=Diffusers_SD.dpmpp_2m_karras, bs=4, width=1024, height=1024,
             seed=42, N_steps=20, guidance_scale=7.0, save_root='output_pipe/'):
    return dict(workflow=Actions(actions=[
        build_model_SDXL(pretrained_model=pretrained_model, noise_sampler=noise_sampler),
        optimize_model(),
        text_SDXL(prompt=prompt, negative_prompt=negative_prompt, bs=bs),
        config_diffusion(width=width, height=height, seed=seed, N_steps=N_steps),
        diffusion(guidance_scale=guidance_scale),
        decode(save_root=save_root)
    ]))