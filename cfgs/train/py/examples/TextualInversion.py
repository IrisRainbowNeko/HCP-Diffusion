from cfgs.train.py import train_base, tuning_base
from hcpdiff.ckpt_manager.format import DiffusersSD15Format
from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.data import VaeCache
from hcpdiff.easy import SD15_auto_loader
from hcpdiff.models import SD15Wrapper
from rainbowneko.ckpt_manager import ckpt_saver, LAYERS_TRAINABLE, LocalCkptSource
from rainbowneko.parser import neko_cfg
from rainbowneko.data import RatioBucket
from hcpdiff.parser import CfgEmbPTParser
from hcpdiff.evaluate import HCPPreviewer

from cfgs.workflow import t2i_TextualInversion
# replace the prompt and negative_prompt
t2i_TextualInversion.prompt = ('pt-paimeng, 1girl, halo, white_hair, solo, smile, blue_eyes, looking_at_viewer, open_mouth, long_sleeves, white_dress, dress, single_thighhigh,'
          ' :d, cape, hair_between_eyes, thighhighs, hair_ornament, blush, white_outline, outline, sky, scarf, cloud, white_thighhighs, arm_up,'
          ' notice_lines, paimon_(genshin_impact)')
t2i_TextualInversion.negative_prompt = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality,'
                   ' normal quality, jpeg artifacts, signature, watermark, username, blurry')

@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],
        mixed_precision='fp16',

        emb_pt=CfgEmbPTParser(
            emb_dir='embs/',
            cfg_pt={
                'pt-paimeng': dict(lr=0.003, weight_decay=1e-2)
            }
        ),

        ckpt_saver=dict(
            SD15=ckpt_saver(
                ckpt_type='safetensors',
                target_module='denoiser',
                layers=LAYERS_TRAINABLE,
            )
        ),

        train=dict(
            train_steps=1000,
            save_step=100,
        ),

        model=dict(
            name='model',

            ## Easy config
            wrapper=SD15Wrapper.from_pretrained(
                _partial_=True,
                models=SD15_auto_loader(ckpt_path='Lykon/DreamShaper', _partial_=True),
            ),
        ),

        data_train=cfg_data(),
        evaluator=HCPPreviewer(_partial_=True,
            interval=100,
            workflow=t2i_TextualInversion,
        ),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=TextImagePairDataset(_partial_=True, batch_size=4, loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root= 'imgs/',
                    label_file= '${.img_root}',  # path to image captions
                    prompt_template='prompt_template/object.txt',
                ),
            ),
            handler=StableDiffusionHandler(
                bucket=RatioBucket,
                word_names=dict(
                    pt1='pt-paimeng'
                )
                ),
            bucket=RatioBucket.from_files(
                target_area=512*512,
                num_bucket=6,
            ),
            cache=VaeCache(bs=1)
        )
    )