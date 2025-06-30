import torch
from rainbowneko.infer import LoadModelAction, BuildPluginAction
from rainbowneko.infer.workflow import Actions, PrepareAction
from rainbowneko.parser import neko_cfg

from cfgs.workflow.text2img import *
from hcpdiff.easy import SD15_auto_loader, Diffusers_SD
from hcpdiff.ckpt_manager import HCPLoraLoader
from hcpdiff.workflow import BuildModelsAction

prompt = ('paimeng, 1girl, halo, white_hair, solo, smile, blue_eyes, looking_at_viewer, open_mouth, long_sleeves, white_dress, dress, single_thighhigh,'
          ' :d, cape, hair_between_eyes, thighhighs, hair_ornament, blush, white_outline, outline, sky, scarf, cloud, white_thighhighs, arm_up,'
          ' notice_lines, paimon_(genshin_impact)')
negative_prompt = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality,'
                   ' normal quality, jpeg artifacts, signature, watermark, username, blurry')

@neko_cfg
def load_lora() -> Actions:
    return Actions([
        LoadModelAction(cfg=dict(
            lora1=HCPLoraLoader(
                path='exps/lora_paimeng/ckpts/lora_unet-1000.safetensors',
                state_prefix='denoiser.',
                alpha=1,
            )
        ), key_map_in=('denoiser -> model', 'in_preview -> in_preview'))
    ])

@neko_cfg
def make_cfg(pretrained_model='Lykon/DreamShaper'):
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model=pretrained_model),
        load_lora(),
        optimize_model(),
        text(prompt=prompt, negative_prompt=negative_prompt),
        config_diffusion(),
        diffusion(),
        decode()
    ]))
