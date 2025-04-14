from hcpdiff.data.handler import ControlNetHandler, StableDiffusionHandler
from hcpdiff.models import ControlNetPlugin
from rainbowneko.data import SyncHandler
from rainbowneko.parser import neko_cfg

@neko_cfg
def ControlNet_SD15(lr=1e-4):
    return ControlNetPlugin(
        _partial_=True,
        lr=lr,
        from_layers=[
            'pre_hook:',
            'pre_hook:conv_in',  # to make forward inside autocast
        ],
        to_layers=[
            'down_blocks.0',
            'down_blocks.1',
            'down_blocks.2',
            'down_blocks.3',
            'mid_block',
            'pre_hook:up_blocks.3.resnets.2',
        ]
    )

@neko_cfg
def make_controlnet_handler(bucket=None, encoder_attention_mask=False, erase=0.15, dropout=0.0, shuffle=0.0, word_names={}):
    return SyncHandler(
        diffusion=StableDiffusionHandler(bucket=bucket, encoder_attention_mask=encoder_attention_mask, erase=erase, dropout=dropout, shuffle=shuffle,
                                         word_names=word_names),
        cnet=ControlNetHandler(bucket=bucket)
    )
