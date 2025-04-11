import torch
from rainbowneko.infer.workflow import Actions, PrepareAction
from rainbowneko.parser import neko_cfg

from cfgs.workflow.text2img import *
from hcpdiff.easy import SD15_auto_loader, Diffusers_SD
from hcpdiff.workflow import BuildModelsAction

prompt = ('pt-paimeng-500, 1girl, halo, white_hair, solo, smile, blue_eyes, looking_at_viewer, open_mouth, long_sleeves, white_dress, dress, single_thighhigh,'
          ' :d, cape, hair_between_eyes, thighhighs, hair_ornament, blush, white_outline, outline, sky, scarf, cloud, white_thighhighs, arm_up,'
          ' notice_lines, paimon_(genshin_impact)')
negative_prompt = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality,'
                   ' normal quality, jpeg artifacts, signature, watermark, username, blurry')

@neko_cfg
def build_model(pretrained_model='ckpts/any5') -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=torch.float16),
        ## Full config
        # BuildModelsAction(
        #     model_loader=ModelManager(
        #         source=LocalCkptSource(),
        #         format=DiffusersSD15Format()
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
            model_loader=SD15_auto_loader(_partial_=True,
                                          ckpt_path=pretrained_model,
                                          noise_sampler=Diffusers_SD.dpmpp_2m_karras
                                          )
        ),
    ])

@neko_cfg
def text(bs=4) -> Actions:
    return Actions([
        TextHookAction(N_repeats=1, layer_skip=1, emb_dir='/mnt/SSD_3TB/dzy/HCP-Diffusion/exps/TI_paimeng/ckpts/'),
        AttnMultTextEncodeAction(
            prompt=prompt,
            negative_prompt=negative_prompt,
            bs=bs
        ),
    ])

@neko_cfg
def make_cfg():
    dict(workflow=Actions(actions=[
        build_model(pretrained_model='/mnt/SSD_3TB/dzy/models/DreamShaper'),
        optimize_model(),
        text(),
        config_diffusion(),
        diffusion(),
        decode()
    ]))
