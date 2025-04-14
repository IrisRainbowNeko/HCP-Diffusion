from rainbowneko.ckpt_manager import NekoResumer, NekoModelLoader
from rainbowneko.ckpt_manager import ckpt_saver, LAYERS_TRAINABLE
from rainbowneko.parser import neko_cfg

from cfgs.train.py.examples import SD_FT

@neko_cfg
def make_cfg():
    return dict(
        _base_=[SD_FT],
        mixed_precision='fp16',

        ckpt_saver=dict(
            SD15=ckpt_saver(
                ckpt_type='safetensors',
                target_module='denoiser',
                layers=LAYERS_TRAINABLE,
            )
        ),

        train=dict(
            resume=NekoResumer(
                start_step=1000,
                loader=dict(  # same as workflow
                    model=NekoModelLoader(
                        path='path/to/model',
                        target_module='denoiser',
                    )
                )
            ),
        ),
    )
