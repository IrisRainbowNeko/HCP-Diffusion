from cfgs.workflow.text2img import *
from hcpdiff.workflow import LatentResizeAction

@neko_cfg
def resize():
    return Actions([
        LatentResizeAction(width=1024, height=1024)
    ])

@neko_cfg
def config_highres():
    return Actions([
        SeedAction(42),
        MakeTimestepsAction(N_steps=20, strength=0.6),
        MakeLatentAction(width=1024, height=1024)
    ])

@neko_cfg
def make_cfg():
    return ict(workflow=Actions(actions=[
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