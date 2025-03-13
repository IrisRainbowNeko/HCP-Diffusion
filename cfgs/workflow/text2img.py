import torch
from hcpdiff.ckpt_manager import DiffusersSD15Format
from hcpdiff.workflow import (BuildModelsAction, PrepareDiffusionAction, XformersEnableAction, VaeOptimizeAction, TextHookAction,
                              AttnMultTextEncodeAction, SeedAction, MakeTimestepsAction, MakeLatentAction, DiffusionStepAction, time_iter,
                              DecodeAction, SaveImageAction)
from rainbowneko.ckpt_manager import ModelManager, LocalCkptSource
from rainbowneko.infer.workflow import (Actions, PrepareAction, LoopAction)
from rainbowneko.utils import neko_cfg
from diffusers import DPMSolverMultistepScheduler
from hcpdiff.easy import Diffusers_SD, sd15_auto_loader

## Full config
# @neko_cfg
# def build_model(pretrained_model='ckpts/any5') -> Actions:
#     Actions([
#         PrepareAction(device='cuda', dtype=torch.float16),
#         BuildModelsAction(
#             model_loader=ModelManager(
#                 source=LocalCkptSource(),
#                 format=DiffusersSD15Format()
#             ).load(_partial_=True, name=pretrained_model,
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
def build_model(pretrained_model='ckpts/any5') -> Actions:
    Actions([
        PrepareAction(device='cuda', dtype=torch.float16),
        BuildModelsAction(
            model_loader=sd15_auto_loader(_partial_=True,
                ckpt_path=pretrained_model,
                noise_sampler=Diffusers_SD.dpmpp_2m_karras
            )
        ),
    ])

@neko_cfg
def optimize_model() -> Actions:
    Actions([
        PrepareDiffusionAction(),
        XformersEnableAction(),
        VaeOptimizeAction(slicing=True),
    ])

@neko_cfg
def text(bs=4) -> Actions:
    Actions([
        TextHookAction(N_repeats=1, layer_skip=1),
        XformersEnableAction(),
        AttnMultTextEncodeAction(
            prompt='masterpiece, best quality, 1girl, cat ears, outside',
            negative_prompt='lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
            bs=bs
        ),
    ])

@neko_cfg
def config_diffusion() -> Actions:
    Actions([
        SeedAction(42),
        MakeTimestepsAction(N_steps=20),
        MakeLatentAction(width=512, height=512)
    ])

@neko_cfg
def diffusion() -> Actions:
    Actions([
        LoopAction(
            iterator=time_iter,
            actions=[
                DiffusionStepAction(guidance_scale=7.0)
            ]
        )
    ])

@neko_cfg
def decode() -> Actions:
    Actions([
        DecodeAction(),
        SaveImageAction(save_root='output_pipe/', image_type='png'),
    ])

def make_cfg():
    Actions(actions=[
        build_model(pretrained_model='/mnt/SSD_3TB/dzy/models/DreamShaper'),
        optimize_model(),
        text(),
        config_diffusion(),
        diffusion(),
        decode()
    ])
