from cfgs.train.py.examples import SD_FT
from rainbowneko.parser import CfgWDPluginParser
from rainbowneko.ckpt_manager import ckpt_manager
from rainbowneko.utils import neko_cfg
from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.models import StableDiffusionWrapper
from rainbowneko.train.data import RatioBucket
from hcpdiff.data import VaeCache
from hcpdiff.models.lora_layers_patch import LoraLayer

def make_cfg():
    dict(
        _base_=[SD_FT],
        mixed_precision='fp16',

        model_part=None,
        model_plugin=CfgWDPluginParser(cfg_plugin=dict(
            lora1=LoraLayer.wrap_model(
                _partial_=True,
                rank=4,
                layers=[
                    're:.*\.attn.?$',
                    're:.*\.ff$',
                ]
            )
        )),

        train=dict(
            train_steps=1000,
            save_step=200,
        ),

        model=dict(
            name='model',

            wrapper=StableDiffusionWrapper.from_pretrained(
                _partial_=True,
                pretrained_model='Lykon/DreamShaper'
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
            handler=StableDiffusionHandler(RatioBucket),
            bucket=RatioBucket.from_files(
                target_area=512*512,
                num_bucket=4,
            ),
            cache=VaeCache(bs=1)
        )
    )