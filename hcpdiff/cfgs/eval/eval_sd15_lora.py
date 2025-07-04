from cfgs.eval.eval_sd15 import *
from rainbowneko.evaluate import WorkflowEvaluator
from rainbowneko.infer import LoadModelAction
from rainbowneko.loggers import CLILogger
from hcpdiff.ckpt_manager import HCPLoraLoader

from hcpdiff.workflow import TextHookAction

negative_prompt = 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'

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
def workflow():
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model='/data3/dzy/models/MeinaMix_V11'),
        load_lora(),
        optimize_model(),
        TextHookAction(N_repeats=1, layer_skip=1),
        generate_from_dataset(data_root='/data3/dzy/dataset/paimeng/paimeng', negative_prompt=negative_prompt)
    ]))

@neko_cfg
def make_cfg():
    return WorkflowEvaluator(
        _partial_=True,
        exp_dir=f'exps_eval/lora_paimeng',
        mixed_precision='fp16',
        seed=42,

        logger=[
            CLILogger(_partial_=True, out_path='train.log', log_step=20),
        ],

        workflow=workflow()
    )