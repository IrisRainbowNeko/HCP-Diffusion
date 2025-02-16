from cfgs.py.train import train_base, tuning_base
from rainbowneko.parser import CfgWDModelParser
from rainbowneko.ckpt_manager import ckpt_manager
from rainbowneko.utils import neko_cfg
from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.models import StableDiffusionWrapper
from rainbowneko.train.data import RatioBucket

def make_cfg():
    dict(
        _base_=[train_base, tuning_base],

        model_part=CfgWDModelParser([
            dict(
                lr=1e-5,
                layers=['unet'],  # train UNet
            )
        ]),

        ckpt_manager=[
            ckpt_manager('safetensors', saved_model=({'model':'unet', 'trainable':False},))
        ],

        train=dict(
            train_steps=5000,
            save_step=1000,
        ),

        model=dict(
            name='model',

            wrapper=StableDiffusionWrapper.from_pretrained(
                _partial_=True,
                pretrained_model='runwayml/stable-diffusion-v1-5'
            ),
        ),

        data_train=cfg_data(),
    )

@neko_cfg
def cfg_data():
    dict(
        dataset1=TextImagePairDataset(_partial_=True, batch_size=8, loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root= 'imgs/',
                    label_file= '',  # path to image captions (file_words)
                    prompt_template='prompt_tuning_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(RatioBucket),
            bucket=RatioBucket.from_files(
                target_area=512*512,
                num_bucket=6,
            ),
        )
    )