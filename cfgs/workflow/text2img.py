import torch
from hcpdiff.ckpt_manager import DiffusersSD15Format
from hcpdiff.workflow import (BuildModelsAction, PrepareDiffusionAction, XformersEnableAction, VaeOptimizeAction, TextHookAction,
                              AttnMultTextEncodeAction, SeedAction, MakeTimestepsAction, MakeLatentAction, DiffusionStepAction, time_iter,
                              DecodeAction, SaveImageAction)
from rainbowneko.ckpt_manager import ModelManager, LocalCkptSource
from rainbowneko.infer.workflow import (Actions, PrepareAction, LoopAction)
from rainbowneko.utils import neko_cfg

@neko_cfg
def build_model(pretrained_model='ckpts/any5') -> Actions:
    Actions([
        PrepareAction(device='cpu', dtype=torch.float16),
        BuildModelsAction(
            model_manager=ModelManager(
                source=LocalCkptSource(),
                format=DiffusersSD15Format()
            ).load(_partial_=True, name=pretrained_model)
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
        build_model(),
        optimize_model(),
        text(),
        config_diffusion(),
        diffusion(),
        decode()
    ])
