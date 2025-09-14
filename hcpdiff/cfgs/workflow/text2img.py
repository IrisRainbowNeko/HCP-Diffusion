import torch
from hcpdiff.ckpt_manager import DiffusersSD15Format
from hcpdiff.workflow import (BuildModelsAction, PrepareDiffusionAction, XformersEnableAction, VaeOptimizeAction, TextHookAction,
                              AttnMultTextEncodeAction, SeedAction, MakeTimestepsAction, MakeLatentAction, DiffusionStepAction, time_iter,
                              DecodeAction, SaveImageAction)
from rainbowneko.ckpt_manager import NekoLoader, LocalCkptSource
from rainbowneko.infer.workflow import (Actions, PrepareAction, LoopAction)
from rainbowneko.parser import neko_cfg
from diffusers import DPMSolverMultistepScheduler
from hcpdiff.easy import Diffusers_SD, SD15_auto_loader

prompt='masterpiece, best quality, 1girl, cat ears, outside'
negative_prompt = 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'

## Full config
# @neko_cfg
# def build_model(pretrained_model='ckpts/any5') -> Actions:
#     return Actions([
#         PrepareAction(device='cuda', dtype=torch.float16),
#         BuildModelsAction(
#             model_loader=NekoLoader(
#                 source=LocalCkptSource(),
#                 format=DiffusersSD15Format()
#             ).load(_partial_=True, path=pretrained_model,
#                 noise_sampler=DPMSolverMultistepScheduler(
#                     beta_start=0.00085,
#                     beta_end=0.012,
#                     beta_schedule='scaled_linear',
#                     algorithm_type='sde-dpmsolver++',
#                     use_karras_sigmas=True,
#                 )
#             )
#         ),
#     ])

## Easy config
@neko_cfg
def build_model(pretrained_model='ckpts/any5', noise_sampler=Diffusers_SD.dpmpp_2m_karras) -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=torch.float16),
        BuildModelsAction(
            model_loader=SD15_auto_loader(_partial_=True,
                ckpt_path=pretrained_model,
                noise_sampler=noise_sampler
            )
        ),
    ])

@neko_cfg
def optimize_model(amp=torch.float16) -> Actions:
    return Actions([
        PrepareDiffusionAction(amp=amp, model_offload=True),
        XformersEnableAction(),
        VaeOptimizeAction(slicing=True),
    ])

@neko_cfg
def text(prompt=prompt, negative_prompt=negative_prompt, bs=4, N_repeats=1, layer_skip=1, TE_final_norm=True) -> Actions:
    return Actions([
        TextHookAction(N_repeats=N_repeats, layer_skip=layer_skip, TE_final_norm=TE_final_norm),
        AttnMultTextEncodeAction(
            prompt=prompt,
            negative_prompt=negative_prompt,
            bs=bs
        ),
    ])

@neko_cfg
def config_diffusion(seed=42, N_steps=20, width=512, height=512) -> Actions:
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
def make_cfg(pretrained_model='Lykon/DreamShaper', prompt=prompt, negative_prompt=negative_prompt):
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model=pretrained_model),
        optimize_model(),
        text(prompt=prompt, negative_prompt=negative_prompt),
        config_diffusion(),
        diffusion(),
        decode()
    ]))
