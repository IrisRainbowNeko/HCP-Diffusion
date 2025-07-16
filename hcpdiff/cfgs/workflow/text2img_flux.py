import torch
from rainbowneko.infer.workflow import Actions, PrepareAction
from rainbowneko.parser import neko_cfg

from cfgs.workflow.text2img import *
from hcpdiff.easy import Flux_auto_loader, Diffusers_SD
from hcpdiff.workflow import BuildModelsAction, TextHookAction, AttnMultTextEncodeAction, BuildOffloadAction, MakeLatentAction

prompt = ('paimeng, 1girl, halo, white_hair, solo, smile, blue_eyes, looking_at_viewer, open_mouth, long_sleeves, white_dress, dress, single_thighhigh,'
          ' :d, cape, hair_between_eyes, thighhighs, hair_ornament, blush, white_outline, outline, sky, scarf, cloud, white_thighhighs, arm_up,'
          ' notice_lines, paimon_(genshin_impact)')
negative_prompt = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality,'
                   ' normal quality, jpeg artifacts, signature, watermark, username, blurry')

@neko_cfg
def build_model(pretrained_model='ckpts/any5') -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=torch.float8_e4m3fn),
        ## Easy config
        BuildModelsAction(
            model_loader=Flux_auto_loader(
                _partial_=True,
                ckpt_path=pretrained_model,
                noise_sampler=Diffusers_SD.euler_flow
            )
        ),
    ])

@neko_cfg
def optimize_model() -> Actions:
    return Actions([
        PrepareDiffusionAction(),
        XformersEnableAction(),
        VaeOptimizeAction(slicing=True),
        BuildOffloadAction(max_VRAM='16GiB', max_RAM='64GiB'),
    ])

@neko_cfg
def text(bs=4) -> Actions:
    return Actions([
        TextHookAction(N_repeats=1),
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
        MakeLatentAction(N_ch=16, width=width, height=height, patch_size=2)
    ])

@neko_cfg
def make_cfg(pretrained_model='black-forest-labs/FLUX.1-dev'):
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model=pretrained_model),
        optimize_model(),
        text(),
        config_diffusion(width=1024, height=1024),
        diffusion(),
        decode()
    ]))
