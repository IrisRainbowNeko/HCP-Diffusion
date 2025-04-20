import torch
from rainbowneko.data import RatioBucket
from rainbowneko.parser import CfgWDPluginParser, neko_cfg
from rainbowneko.utils import ConstantLR
from rainbowneko.ckpt_manager import plugin_saver

from cfgs.train.py.examples import SD_FT
from hcpdiff.data import TextImagePairDataset, Text2ImageCondSource
from hcpdiff.data import VaeCache
from hcpdiff.easy import SD15_auto_loader, ControlNet_SD15, make_controlnet_handler
from hcpdiff.models import SD15Wrapper

@neko_cfg
def make_cfg():
    return dict(
        _base_=[SD_FT],
        mixed_precision='fp16',

        model_part=None,
        model_plugin=CfgWDPluginParser(cfg_plugin=dict(
            cnet=ControlNet_SD15(lr=1e-4)
        ), weight_decay=1e-2),

        ckpt_saver=dict(
            cnet=plugin_saver(
                ckpt_type='safetensors',
                target_plugin='cnet',
            )
        ),

        train=dict(
            train_steps=10000,
            save_step=2000,

            optimizer=torch.optim.AdamW(_partial_=True),

            lr_scheduler=ConstantLR(
                _partial_=True,
                warmup_steps=1000,
            ),
        ),

        model=dict(
            name='model',

            wrapper=SD15Wrapper.from_pretrained(
                models=SD15_auto_loader(ckpt_path='Lykon/DreamShaper', _partial_=True),
                _partial_=True,
            ),
        ),

        data_train=cfg_data(),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=TextImagePairDataset(_partial_=True, batch_size=4, loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageCondSource(  # NOTE: source for control
                    img_root='imgs/',
                    cond_dir='conds/',
                    label_file='${.img_root}',  # path to image captions (file_words)
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=make_controlnet_handler(bucket=RatioBucket),
            bucket=RatioBucket.from_files(
                target_area=512*512,
                num_bucket=4,
            ),
            cache=VaeCache(bs=1)
        )
    )
