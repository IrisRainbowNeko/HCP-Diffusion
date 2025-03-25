from cfgs.workflow.text2img import *
from rainbowneko.infer.workflow import Actions, PrepareAction
from hcpdiff.workflow import BuildModelsAction
import torch
from rainbowneko.ckpt_manager import ModelManager, LocalCkptSource
from hcpdiff.ckpt_manager import DiffusersSDXLFormat
from diffusers import DPMSolverMultistepScheduler
from rainbowneko.utils import neko_cfg
from rainbowneko.infer import BuildPluginAction, LoadModelAction
from rainbowneko.parser import CfgWDPluginParser
from hcpdiff.models.lora_layers_patch import LoraLayer
from hcpdiff.easy import SDXL_auto_loader, Diffusers_SD
from hcpdiff.parser import HCPLoraLoader

prompt = ('paimeng, 1girl, halo, white_hair, solo, smile, blue_eyes, looking_at_viewer, open_mouth, long_sleeves, white_dress, dress, single_thighhigh,'
          ' :d, cape, hair_between_eyes, thighhighs, hair_ornament, blush, white_outline, outline, sky, scarf, cloud, white_thighhighs, arm_up,'
          ' notice_lines, paimon_(genshin_impact)')
negative_prompt = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality,'
                   ' normal quality, jpeg artifacts, signature, watermark, username, blurry')

@neko_cfg
def build_model(pretrained_model='ckpts/any5') -> Actions:
    Actions([
        PrepareAction(device='cuda', dtype=torch.float16),
        ## Full config
        # BuildModelsAction(
        #     model_loader=ModelManager(
        #         source=LocalCkptSource(),
        #         format=DiffusersSDXLFormat()
        #     ).load(_partial_=True, name=pretrained_model,
        #         noise_sampler=DPMSolverMultistepScheduler(
        #             beta_start=0.00085,
        #             beta_end=0.012,
        #             beta_schedule='scaled_linear',
        #             algorithm_type='sde-dpmsolver++',
        #             use_karras_sigmas=True,
        #         )
        #     )
        # ),
        ## Easy config
        BuildModelsAction(
            model_loader=SDXL_auto_loader(_partial_=True,
                                          ckpt_path=pretrained_model,
                                          noise_sampler=Diffusers_SD.dpmpp_2m_karras
                                          )
        ),
        LoadModelAction(cfg=dict(
            lora1=HCPLoraLoader(
                path='exps/lora_sdxl_paimeng/ckpts/model-1000-lora1.safetensors',
                alpha=2,
            )
        ), key_map_in=('denoiser -> model', 'in_preview -> in_preview'))
    ])

@neko_cfg
def text(bs=4) -> Actions:
    Actions([
        TextHookAction(N_repeats=1, layer_skip=1, TE_final_norm=False),
        AttnMultTextEncodeAction(
            prompt=prompt,
            negative_prompt=negative_prompt,
            bs=bs
        ),
    ])

@neko_cfg
def config_diffusion() -> Actions:
    Actions([
        SeedAction(42),
        MakeTimestepsAction(N_steps=20),
        MakeLatentAction(width=1024, height=1024)
    ])

def make_cfg():
    dict(workflow=Actions(actions=[
        build_model(pretrained_model='/mnt/SSD_3TB/dzy/models/Illustrious-XL-v1.1/Illustrious-XL-v1.1.safetensors'),
        optimize_model(),
        text(),
        config_diffusion(),
        diffusion(),
        decode()
    ]))
