import torch
from hcpdiff.ckpt_manager import LoraWebuiFormat
from hcpdiff.easy import SDXL_auto_loader
from hcpdiff.models import SDXLWrapper
from hcpdiff.models.lora_layers_patch import LoraLayer
from rainbowneko.ckpt_manager import ckpt_saver, NekoPluginSaver, LAYERS_TRAINABLE, SafeTensorFormat, NekoOptimizerSaver
from rainbowneko.parser import CfgWDPluginParser, neko_cfg, CfgWDModelParser, disable_neko_cfg
from rainbowneko.utils import ConstantLR

@neko_cfg
def SDXL_finetuning(base_model: str, train_steps: int, dataset, save_step: int = 500, save_optimizer=False, lr: float = 1e-5,
                    dtype: str = 'fp16', low_vram: bool = False, warmup_steps: int = 0, name: str = 'SDXL'):
    if low_vram:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(_partial_=True)
    else:
        optimizer = torch.optim.AdamW(_partial_=True)

    ckpt_saver_dict = dict(
        SDXL=ckpt_saver(
            ckpt_type='safetensors',
            target_module='denoiser',
            layers=LAYERS_TRAINABLE,
        )
    )

    if save_optimizer:
        ckpt_saver_dict['optimizer'] = NekoOptimizerSaver()

    from cfgs.train.py import train_base, tuning_base

    return dict(
        _base_=[train_base, tuning_base],
        mixed_precision=dtype,

        model_part=CfgWDModelParser([
            dict(
                lr=lr,
                layers=['denoiser'],  # train UNet
            )
        ], weight_decay=1e-2),

        ckpt_saver=ckpt_saver_dict,

        train=dict(
            train_steps=train_steps,
            save_step=save_step,

            optimizer=optimizer,

            lr_scheduler=ConstantLR(
                _partial_=True,
                warmup_steps=warmup_steps,
            ),
        ),

        model=dict(
            name=name,

            ## Easy config
            wrapper=SDXLWrapper.from_pretrained(
                _partial_=True,
                models=SDXL_auto_loader(ckpt_path=base_model, _partial_=True),
            ),
        ),

        data_train=dataset,
    )

@neko_cfg
def SDXL_lora_train(base_model: str, train_steps: int, dataset, save_step: int = 200, save_optimizer=False, lr: float = 1e-4, rank: int = 4,
                    alpha: float = None, with_conv: bool = False, dtype: str = 'fp16', low_vram: bool = False, warmup_steps: int = 0,
                    name: str = 'SDXL', save_webui_format=False):
    with disable_neko_cfg:
        if alpha is None:
            alpha = rank

        if with_conv:
            lora_layers = [
                r're:denoiser.*\.attn.?$',
                r're:denoiser.*\.ff$',
                r're:denoiser.*\.resnets$',
                r're:denoiser.*\.proj_in$',
                r're:denoiser.*\.proj_out$',
                r're:denoiser.*\.conv$',
            ]
        else:
            lora_layers = [
                r're:denoiser.*\.attn.?$',
                r're:denoiser.*\.ff$',
            ]

    if low_vram:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(_partial_=True, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.AdamW(_partial_=True, betas=(0.9, 0.99))

    if save_webui_format:
        lora_format = LoraWebuiFormat()
    else:
        lora_format = SafeTensorFormat()

    ckpt_saver_dict = dict(
        _replace_=True,
        lora_unet=NekoPluginSaver(
            format=lora_format,
            target_plugin='lora1',
        )
    )

    if save_optimizer:
        ckpt_saver_dict['optimizer'] = NekoOptimizerSaver()

    from cfgs.train.py.examples import SD_FT

    return dict(
        _base_=[SD_FT],
        mixed_precision=dtype,

        model_part=None,
        model_plugin=CfgWDPluginParser(cfg_plugin=dict(
            lora1=LoraLayer.wrap_model(
                _partial_=True,
                lr=lr,
                rank=rank,
                alpha=alpha,
                layers=lora_layers
            )
        ), weight_decay=0.1),

        ckpt_saver=ckpt_saver_dict,

        train=dict(
            train_steps=train_steps,
            save_step=save_step,

            optimizer=optimizer,

            lr_scheduler=ConstantLR(
                _partial_=True,
                warmup_steps=warmup_steps,
            ),
        ),

        model=dict(
            name=name,

            wrapper=SDXLWrapper.from_pretrained(
                models=SDXL_auto_loader(ckpt_path=base_model, _partial_=True),
                _partial_=True,
            ),
        ),

        data_train=dataset,
    )
