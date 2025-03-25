from bitsandbytes.optim import AdamW8bit
from rainbowneko.data import RatioBucket
from rainbowneko.parser import CfgWDModelParser
from rainbowneko.utils import ConstantLR
from rainbowneko.utils import neko_cfg

from cfgs.train.py.examples import SD_FT
from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.data import VaeCache
from hcpdiff.easy import SDXL_auto_loader
from hcpdiff.models import SDXLWrapper

def make_cfg():
    dict(
        _base_=[SD_FT],
        mixed_precision='fp16',

        model_part=CfgWDModelParser([
            dict(
                lr=1e-5,
                layers=['denoiser'],  # train UNet
            )
        ], weight_decay=1e-2),

        train=dict(
            train_steps=1000,
            save_step=200,

            optimizer=AdamW8bit(_partial_=True),

            scheduler=ConstantLR(
                _partial_=True,
                warmup_steps=100,
            ),
        ),

        model=dict(
            name='model',

            wrapper=SDXLWrapper.from_pretrained(
                models=SDXL_auto_loader(ckpt_path='stabilityai/stable-diffusion-xl-base-1.0', _partial_=True),
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
                    label_file= '${.img_root}',  # path to image captions (file_words)
                    prompt_template='prompt_tuning_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(bucket=RatioBucket),
            bucket=RatioBucket.from_files(
                target_area=1024*1024,
                num_bucket=6,
            ),
            cache=VaeCache(bs=1)
        )
    )