from cfgs.workflow.text2img import *
from hcpdiff.workflow import LoadImageAction, EncodeAction

@neko_cfg
def config_diffusion() -> Actions:
    return Actions([
        SeedAction(42),
        MakeTimestepsAction(N_steps=20, strength=0.6),
        LoadImageAction(image_paths='cond.png'),
        EncodeAction(),
        MakeLatentAction(width=512, height=512)
    ])

@neko_cfg
def make_cfg():
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model='/mnt/SSD_3TB/dzy/models/DreamShaper'),
        optimize_model(),
        text(),
        config_diffusion(),
        diffusion(),
        decode()
    ]))
