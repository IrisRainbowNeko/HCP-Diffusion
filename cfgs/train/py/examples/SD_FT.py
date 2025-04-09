import torch
from cfgs.train.py import train_base, tuning_base
from hcpdiff.ckpt_manager.format import DiffusersSD15Format
from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.data import VaeCache
from hcpdiff.easy import SD15_auto_loader
from hcpdiff.models import SD15Wrapper
from rainbowneko.ckpt_manager import ckpt_saver, NekoLoader, LocalCkptSource, LAYERS_TRAINABLE
from rainbowneko.parser import CfgWDModelParser, neko_cfg
from rainbowneko.data import RatioBucket
from rainbowneko.utils import ConstantLR

@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],
        mixed_precision='fp16',

        model_part=CfgWDModelParser([
            dict(
                lr=1e-5,
                layers=['denoiser'],  # train UNet
            )
        ], weight_decay=1e-2),

        ckpt_saver=dict(
            SD15=ckpt_saver(
                ckpt_type='safetensors',
                target_module='denoiser',
                layers=LAYERS_TRAINABLE,
            )
        ),

        train=dict(
            train_steps=5000,
            save_step=500,

            optimizer=torch.optim.AdamW(_partial_=True),

            scheduler=ConstantLR(
                _partial_=True,
                warmup_steps=500,
            ),
        ),

        model=dict(
            name='SD15',

            ## Full config
            # wrapper=SD15Wrapper.from_pretrained(
            #     _partial_=True,
            #     models=NekoLoader(
            #         format=DiffusersSD15Format(),
            #         source=LocalCkptSource(),
            #     ).load(path='Lykon/DreamShaper', _partial_=True)
            # ),

            ## Easy config
            wrapper=SD15Wrapper.from_pretrained(
                _partial_=True,
                models=SD15_auto_loader(ckpt_path='Lykon/DreamShaper', _partial_=True),
            ),
        ),

        data_train=cfg_data(),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=TextImagePairDataset(_partial_=True, batch_size=4, loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root= 'imgs/',
                    label_file= '${.img_root}',  # path to image captions
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(bucket=RatioBucket),
            bucket=RatioBucket.from_files(
                target_area=512*512,
                num_bucket=6,
            ),
            cache=VaeCache(bs=1)
        )
    )