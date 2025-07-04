from cfgs.workflow.text2img import *
from rainbowneko.data import FixedBucket, ImageHandler
from torchvision import transforms as T
from rainbowneko.evaluate import MetricContainer, MetricGroup, WorkflowEvaluator
from rainbowneko.loggers import CLILogger
from rainbowneko.utils import KeyMapper
from rainbowneko.infer import DataLoaderAction, LambdaAction, MetricAction, HandlerAction
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from hcpdiff.evaluate import CLIPScoreMetric

from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.workflow import TextHookAction, AttnMultTextEncodeAction, DecodeAction, SaveImageAction, DiffusionActions

@neko_cfg
def cal_metrics(save_root='output_eval/') -> Actions:
    return Actions([
        DecodeAction(),
        SaveImageAction(save_root=save_root, image_type='png', save_cfg=False),
        HandlerAction(
            handler=ImageHandler(transform=T.Compose([
                T.ToTensor(),
                T.Normalize([0.5], [0.5])
            ]),
            key_map_in=('images -> image',),)
        ),
        LambdaAction(f_act=lambda image, **states: {'image': torch.stack(image).to('cuda'), }),
        MetricAction(metric=MetricGroup(
            lpips=MetricContainer(LPIPS().to('cuda'), device='cuda', key_map=('pred -> 0', 'inputs.image -> 1')),
            clip_score=MetricContainer(CLIPScoreMetric(), key_map=('pred -> 0', 'inputs.prompt -> 1')),
        ), key_map_in=('image -> pred', 'prompt -> inputs.prompt', 'image_real -> inputs.image', 'device -> device'))
    ])


@neko_cfg
def generate_from_dataset(data_root='imgs/', negative_prompt=negative_prompt, bs=4, seed=42, N_steps=20,
                           width=512, height=512, guidance_scale=7.0) -> Actions:
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
                    tokenize=False,
                    key_map_out=KeyMapper(key_map=('image -> image_real',), move_mode=True),
                ),
                bucket=FixedBucket(target_size=(width, height)),
            ),
            actions=DiffusionActions([
                AttnMultTextEncodeAction(
                    prompt=None,
                    negative_prompt=negative_prompt,
                    bs=bs
                ),
                config_diffusion(N_steps=N_steps, width=width, height=height),
                diffusion(guidance_scale=guidance_scale),
                cal_metrics(save_root='output_eval/'),
            ])
        )
    ])

@neko_cfg
def workflow():
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model='/mnt/SSD_3TB/dzy/models/DreamShaper'),
        optimize_model(),
        TextHookAction(N_repeats=1, layer_skip=1),
        generate_from_dataset()
    ]))

@neko_cfg
def make_cfg():
    return WorkflowEvaluator(
        _partial_=True,
        exp_dir=f'exps_eval/SD15_paimeng',
        mixed_precision='fp16',
        seed=42,

        logger=[
            CLILogger(_partial_=True, out_path='train.log', log_step=20),
        ],

        workflow=workflow()
    )