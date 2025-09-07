import torch
from hcpdiff.ckpt_manager import LoraWebuiFormat
from hcpdiff.data import TextImagePairDataset, Text2ImageSource, StableDiffusionHandler
from hcpdiff.data import VaeCache
from hcpdiff.easy import SD15_auto_loader
from hcpdiff.models import SD15Wrapper, TEHookCFG
from hcpdiff.models.lora_layers_patch import LoraLayer
from rainbowneko.ckpt_manager import ckpt_saver, NekoOptimizerSaver, LAYERS_TRAINABLE, NekoPluginSaver, SafeTensorFormat
from rainbowneko.data import RatioBucket, FixedBucket
from rainbowneko.parser import CfgWDPluginParser, neko_cfg, CfgWDModelParser, disable_neko_cfg
from rainbowneko.utils import ConstantLR, Path_Like

@neko_cfg
def SD15_finetuning(base_model: str, train_steps: int, dataset, save_step: int = 500, save_optimizer=False, lr: float = 1e-5, clip_skip: int = 0,
                    dtype: str = 'fp16', low_vram: bool = False, warmup_steps: int = 0, name: str = 'SD15'):
    if low_vram:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(_partial_=True)
    else:
        optimizer = torch.optim.AdamW(_partial_=True)

    ckpt_saver_dict = dict(
        SD15=ckpt_saver(
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
            wrapper=SD15Wrapper.from_pretrained(
                _partial_=True,
                models=SD15_auto_loader(ckpt_path=base_model, _partial_=True),
                TE_hook_cfg=TEHookCFG(clip_skip=clip_skip),
            ),
        ),

        data_train=dataset,
    )

@neko_cfg
def SD15_lora_train(base_model: str, train_steps: int, dataset, save_step: int = 200, save_optimizer=False, lr: float = 1e-4, rank: int = 4,
                    alpha: float = None, clip_skip: int = 0, with_conv: bool = False, dtype: str = 'fp16', low_vram: bool = False,
                    warmup_steps: int = 0, name: str = 'SD15', save_webui_format=False):
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

            wrapper=SD15Wrapper.from_pretrained(
                _partial_=True,
                models=SD15_auto_loader(ckpt_path=base_model, _partial_=True),
                TE_hook_cfg=TEHookCFG(clip_skip=clip_skip),
            ),
        ),

        data_train=dataset,
    )

@neko_cfg
def cfg_data_SD_ARB(img_root: Path_Like, batch_size: int = 4, trigger_word: str = '', resolution: int = 512*512, num_bucket=4, word_names=None,
                    prompt_dropout: float = 0, prompt_template: Path_Like = 'prompt_template/caption.txt', loss_weight=1.0):
    if word_names is None:
        word_names = dict(pt1=trigger_word)
    else:
        word_names = word_names

    return TextImagePairDataset(
        _partial_=True, batch_size=batch_size, loss_weight=loss_weight,
        source=dict(
            data_source1=Text2ImageSource(
                img_root=img_root,
                label_file='${.img_root}',  # path to image captions (file_words)
                prompt_template=prompt_template,
            ),
        ),
        handler=StableDiffusionHandler(
            bucket=RatioBucket,
            word_names=word_names,
            erase=prompt_dropout,
        ),
        bucket=RatioBucket.from_files(
            target_area=resolution,
            num_bucket=num_bucket,
        ),
        cache=VaeCache(bs=batch_size)
    )

@neko_cfg
def cfg_data_SD_resize_crop(img_root: Path_Like, batch_size: int = 4, trigger_word: str = '', target_size=(512, 512), word_names=None,
                            prompt_dropout: float = 0, prompt_template: Path_Like = 'prompt_template/caption.txt', loss_weight=1.0):
    if word_names is None:
        word_names = dict(pt1=trigger_word)
    else:
        word_names = word_names

    return TextImagePairDataset(
        _partial_=True, batch_size=batch_size, loss_weight=loss_weight,
        source=dict(
            data_source1=Text2ImageSource(
                img_root=img_root,
                label_file='${.img_root}',  # path to image captions (file_words)
                prompt_template=prompt_template,
            ),
        ),
        handler=StableDiffusionHandler(
            bucket=FixedBucket,
            word_names=word_names,
            erase=prompt_dropout,
        ),
        bucket=FixedBucket(target_size=target_size),
        cache=VaeCache(bs=batch_size)
    )
