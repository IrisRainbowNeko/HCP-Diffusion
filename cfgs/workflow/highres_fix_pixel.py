from cfgs.workflow.text2img import *
from hcpdiff.workflow import ImageResizeAction, DecodeAction, EncodeAction, SaveImageAction

@neko_cfg
def resize():
    Actions([
        DecodeAction(),
        SaveImageAction(save_root='output_pipe/', image_type='webp'),
        ImageResizeAction(width=1024, height=1024, mode='lanczos'),
        EncodeAction(),
    ])

@neko_cfg
def config_highres():
    Actions([
        SeedAction(42),
        MakeTimestepsAction(N_steps=20, strength=0.6),
        MakeLatentAction(width=1024, height=1024)
    ])


def make_cfg():
    dict(workflow=Actions(actions=[
        build_model(pretrained_model='/mnt/SSD_3TB/dzy/models/DreamShaper'),
        optimize_model(),
        text(),
        config_diffusion(),
        diffusion(),
        # >>> highres fix >>>
        resize(),
        config_highres(),
        diffusion(),
        # <<< highres fix <<<
        decode()
    ]))