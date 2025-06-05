from .text2img import *
from rainbowneko.infer import DataLoaderAction, LambdaAction
from rainbowneko.data import BaseBucket
from hcpdiff.data import TextImagePairDataset, TextSource, DiffusionTextHandler

@neko_cfg
def decode(save_root='output_dataset/') -> Actions:
    return Actions([
        DecodeAction(),
        SaveImageAction(save_root=save_root, image_type='png', save_cfg=False, save_txt=True),
    ])

@neko_cfg
def generate_from_dataset(bs=4, seed=42, N_steps=20, width=512, height=512, guidance_scale=7.0) -> Actions:
    return Actions(actions=[
        LambdaAction(f_act=lambda **states: {'seed':seed}),
        DataLoaderAction(
            dataset=TextImagePairDataset(_partial_=True, batch_size=bs,
                source=dict(
                    data_source1=TextSource(label_file='prompts/'),
                ),
                handler=DiffusionTextHandler(),
                bucket=BaseBucket(),
            ),
            actions=Actions([
                AttnMultTextEncodeAction(
                    prompt=None,
                    negative_prompt=None,
                    bs=bs
                ),
                config_diffusion(N_steps=N_steps, width=width, height=height),
                diffusion(guidance_scale=guidance_scale),
                decode(save_root='output_dataset/'),
                LambdaAction(f_act=lambda seed, **states:{'seed':seed+bs}), # different seed for each image
            ])
        )
    ])

@neko_cfg
def make_cfg():
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model='/mnt/SSD_3TB/dzy/models/DreamShaper'),
        optimize_model(),
        TextHookAction(N_repeats=1, layer_skip=1),
        generate_from_dataset()
    ]))
