import torch
from cfgs.train.py.examples import SD_FT
from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.data import VaeCache
from hcpdiff.easy import SD15_auto_loader
from hcpdiff.models import SD15Wrapper
from hcpdiff.models.lora_layers_patch import LoraLayer
from rainbowneko.parser import CfgWDPluginParser
from rainbowneko.train.data import RatioBucket
from rainbowneko.utils import ConstantLR
from rainbowneko.utils import neko_cfg

def make_cfg():
    dict(
        _base_=[SD_FT],
        mixed_precision='fp16',

        model_part=None,
        model_plugin=CfgWDPluginParser(cfg_plugin=dict(
            lora1=LoraLayer.wrap_model(
                _partial_=True,
                lr=1e-4,
                rank=4,
                alpha=2,
                layers=[
                    're:denoiser.*\.attn.?$',
                    're:denoiser.*\.ff$',
                ]
            )
        )),

        train=dict(
            train_steps=1000,
            save_step=200,

            optimizer=torch.optim.AdamW(_partial_=True, betas=(0.9, 0.99), weight_decay=0.1),

            scheduler=ConstantLR(
                _partial_=True,
                warmup_steps=0,
            ),
        ),

        model=dict(
            name='model',

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
                    label_file= '${.img_root}',  # path to image captions (file_words)
                    prompt_template='prompt_tuning_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(RatioBucket, 
                word_names=dict(pt1='paimeng'),
                erase=0,
            ),
            bucket=RatioBucket.from_files(
                target_area=512*512,
                num_bucket=4,
            ),
            cache=VaeCache(bs=1)
        )
    )