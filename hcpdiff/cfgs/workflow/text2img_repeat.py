from rainbowneko.infer import LambdaAction

from cfgs.workflow.text2img import *

@neko_cfg
def diffusion_Ntimes(bs=2, num=2, seed=42, N_steps=20, width=512, height=512, guidance_scale=7.0) -> Actions:
    return Actions(actions=[
        LambdaAction(f_act=lambda **states:{'seed':seed}),
        LoopAction(
            iterator=lambda **states: [{} for i in range(num)],
            actions=[
                config_diffusion(N_steps=N_steps, width=width, height=height),
                diffusion(guidance_scale=guidance_scale),
                decode(),
                LambdaAction(f_act=lambda seed, **states:{'seed':seed+bs}),  # different seed for each image
                LambdaAction(f_act=lambda **states:{'latents':None}),
            ]
        )
    ])

@neko_cfg
def make_cfg():
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model='/mnt/SSD_3TB/dzy/models/DreamShaper'),
        optimize_model(),
        text(),
        diffusion_Ntimes(bs=2, num=2)
    ]))
