import os.path
from typing import Optional, Union

import re
import torch
from torch import nn
from torch.optim import lr_scheduler
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, Optimizer
from omegaconf import OmegaConf
from transformers import PretrainedConfig

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_kwargs = {},
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles (`int`, *optional*):
            The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
        power (`float`, *optional*, defaults to 1.0):
            Power factor. See `POLYNOMIAL` scheduler
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    """
    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    #One Cycle for super convergence
    if name=='one_cycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=[x['lr'] for x in optimizer.state_dict()['param_groups']],
                                steps_per_epoch=num_training_steps, epochs=1,
                                pct_start=num_warmup_steps/num_training_steps, **scheduler_kwargs)
        return scheduler

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **scheduler_kwargs)

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **scheduler_kwargs)

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **scheduler_kwargs
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **scheduler_kwargs
        )

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **scheduler_kwargs)

def low_rank_approximate(weight, rank, clamp_quantile=0.99):
    if len(weight.shape)==4: # conv
        weight=weight.flatten(1)
        out_ch, in_ch, k1, k2 = weight.shape

    U, S, Vh = torch.linalg.svd(weight)
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)

    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, clamp_quantile)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)

    if len(weight.shape) == 4:
        # U is (out_channels, rank) with 1x1 conv.
        U = U.reshape(U.shape[0], U.shape[1], 1, 1)
        # V is (rank, in_channels * kernel_size1 * kernel_size2)
        Vh = Vh.reshape(Vh.shape[0], in_ch, k1, k2)
    return U, Vh

def load_config(path):
    cfg = OmegaConf.load(path)
    if '_base_' in cfg:
        for base in cfg['_base_']:
            cfg = OmegaConf.merge(load_config(base), cfg)
        del cfg['_base_']
    return cfg

def load_config_with_cli(path, args_list=None):
    cfg = load_config(path)
    cfg_cli = OmegaConf.from_cli(args_list)
    cfg = OmegaConf.merge(cfg, cfg_cli)
    return cfg

def _default(v, default):
    return default if v is None else v

def dict_get(data, key, default):
    return data[key] if key in data else default

def var_get(data, key, default):
    return getattr(data, key) if hasattr(data, key) else default

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def cycle_data(data_loader, arb=False):
    epoch=0
    while True:
        if arb:
            data_loader.dataset.bucket.rest(epoch)
        for data in data_loader:
            yield data
        epoch+=1

def get_cfg_range(cfg_text:str):
    dy_cfg_f='ln'
    if cfg_text.find(':')!=-1:
        cfg_text, dy_cfg_f = cfg_text.split(':')

    if cfg_text.find('-')!=-1:
        l, h = cfg_text.split('-')
        return float(l), float(h), dy_cfg_f
    else:
        return float(cfg_text), float(cfg_text), dy_cfg_f

def to_validate_file(name):
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, "_", name)  # 替换为下划线
    return new_title

def make_mask(start, end, length):
    mask=torch.zeros(length)
    mask[int(length*start):int(length*end)]=1
    return mask.bool()

def get_file_name(file: str):
    return file.rsplit('.',1)[0]

def get_file_ext(file: str):
    return file.rsplit('.',1)[1].lower()