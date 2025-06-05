from cfgs.workflow.text2img import *
from rainbowneko.data import FixedBucket, ImageHandler
from torchvision import transforms as T
from rainbowneko.evaluate import MetricContainer, MetricGroup
from rainbowneko.infer import DataLoaderAction, LambdaAction, MetricAction, HandlerAction
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from hcpdiff.evaluate import CLIPScoreMetric

from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.workflow import TextHookAction, AttnMultTextEncodeAction, DecodeAction, SaveImageAction

@neko_cfg
def cal_metrics(save_root='output_eval/') -> Actions:
    return Actions([
        DecodeAction(),
        SaveImageAction(save_root=save_root, image_type='png', save_cfg=False),
        HandlerAction(
            handler=ImageHandler(transform=T.Compose([
                T.ToTensor(),
                T.Normalize([0.5], [0.5])
            ]),)
        ),
        MetricAction(metric=MetricGroup(
            lpips=MetricContainer(LPIPS(), key_map=('pred -> 0', 'inputs.image -> 1')),
            clip_score=MetricContainer(CLIPScoreMetric(), key_map=('pred -> 0', 'inputs.prompt -> 1')),
        ), key_map_in=('image -> pred', 'prompt -> inputs.prompt', 'image_real -> inputs.image', 'device -> device'))
    ])

@neko_cfg
def generate_from_dataset(data_root='imgs/', bs=4, seed=42, N_steps=20, width=512, height=512, guidance_scale=7.0) -> Actions:
    return Actions(actions=[
        LambdaAction(f_act=lambda **states: {'seed':seed}),
        DataLoaderAction(
            dataset=TextImagePairDataset(_partial_=True, batch_size=bs,
                source=dict(
                    data_source1=Text2ImageSource(
                        img_root= data_root,
                        label_file= '${.img_root}',  # path to image captions
                        prompt_template='prompt_template/caption.txt',
                    ),
                ),
                handler=StableDiffusionHandler(
                    bucket=FixedBucket,
                    key_map_out=('image -> image_real', 'coord -> coord', 'prompt -> prompt'),
                ),
                bucket=FixedBucket(target_size=(width, height)),
            ),
            actions=Actions([
                AttnMultTextEncodeAction(
                    prompt=None,
                    negative_prompt=None,
                    bs=bs
                ),
                config_diffusion(N_steps=N_steps, width=width, height=height),
                diffusion(guidance_scale=guidance_scale),
                cal_metrics(save_root='output_eval/'),
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
