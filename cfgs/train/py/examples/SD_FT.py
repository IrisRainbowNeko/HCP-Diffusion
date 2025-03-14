from cfgs.train.py import train_base, tuning_base
from hcpdiff.ckpt_manager.format import DiffusersSD15Format
from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.data import VaeCache
from hcpdiff.easy import SD15_auto_loader
from hcpdiff.models import SD15Wrapper
from rainbowneko.ckpt_manager import ckpt_manager, ModelManager, LocalCkptSource
from rainbowneko.parser import CfgWDModelParser
from rainbowneko.train.data import RatioBucket
from rainbowneko.utils import neko_cfg

def make_cfg():
    dict(
        _base_=[train_base, tuning_base],
        mixed_precision='fp16',

        model_part=CfgWDModelParser([
            dict(
                lr=1e-5,
                layers=['denoiser'],  # train UNet
            )
        ]),

        ckpt_manager=[
            ckpt_manager('safetensors', saved_model=({'model':'denoiser', 'trainable':True},))
        ],

        train=dict(
            train_steps=5000,
            save_step=500,
        ),

        model=dict(
            name='model',

            ## Full config
            # wrapper=StableDiffusionWrapper.from_pretrained(
            #     _partial_=True,
            #     models=ModelManager(
            #         format=DiffusersSD15Format(),
            #         source=LocalCkptSource(),
            #     ).load(name='Lykon/DreamShaper', _partial_=True)
            # ),

            ## Easy config
            wrapper=SD15Wrapper.from_pretrained(
                SD15_auto_loader('Lykon/DreamShaper', _partial_=True),
                _partial_=True,
            ),
        ),

        data_train=cfg_data(),
    )

@neko_cfg
def cfg_data():
    dict(
        dataset1=TextImagePairDataset(_partial_=True, batch_size=4, loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root= 'imgs/',
                    label_file= '${.img_root}',  # path to image captions
                    prompt_template='prompt_tuning_template/caption.txt',
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