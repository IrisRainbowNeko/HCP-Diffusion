import torch
from cfgs.train.py.examples import SD_FT
from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.data import VaeCache
from hcpdiff.easy import SD15_auto_loader
from hcpdiff.models import SD15Wrapper
from hcpdiff.models.lora_layers_patch import LoraLayer
from rainbowneko.parser import CfgWDPluginParser, neko_cfg
from rainbowneko.data import RatioBucket
from rainbowneko.utils import ConstantLR
from rainbowneko.ckpt_manager import plugin_saver
from hcpdiff.evaluate import HCPPreviewer

from cfgs.workflow import t2i_lora
# replace the prompt and negative_prompt in t2i_lora
t2i_lora.prompt = ('paimeng, 1girl, halo, white_hair, solo, smile, blue_eyes, looking_at_viewer, open_mouth, long_sleeves, white_dress, dress, single_thighhigh,'
          ' :d, cape, hair_between_eyes, thighhighs, hair_ornament, blush, white_outline, outline, sky, scarf, cloud, white_thighhighs, arm_up,'
          ' notice_lines, paimon_(genshin_impact)')
t2i_lora.negative_prompt = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality,'
                   ' normal quality, jpeg artifacts, signature, watermark, username, blurry')

@neko_cfg
def make_cfg():
    return dict(
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
        ), weight_decay=0.1),

        ckpt_saver=dict(
            _replace_ = True,
            lora_unet=plugin_saver(
                ckpt_type='safetensors',
                target_plugin='lora1',
            )
        ),

        train=dict(
            train_steps=1000,
            save_step=200,

            optimizer=torch.optim.AdamW(_partial_=True, betas=(0.9, 0.99)),

            lr_scheduler=ConstantLR(
                _partial_=True,
                warmup_steps=0,
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
        evaluator=HCPPreviewer(_partial_=True,
            interval=100,
            workflow=t2i_lora.make_cfg(pretrained_model='${model.wrapper.models.ckpt_path}'),
        ),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=TextImagePairDataset(_partial_=True, batch_size=4, loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root= 'imgs/',
                    label_file= '${.img_root}',  # path to image captions (file_words)
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(
                bucket=RatioBucket, 
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