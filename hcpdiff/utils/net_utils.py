from typing import Optional, Union, Tuple

from torch import nn
from torch.optim import lr_scheduler
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, Optimizer
from transformers import PretrainedConfig

class TEUnetWrapper(nn.Module):
    def __init__(self, unet, TE):
        super().__init__()
        self.unet = unet
        self.TE = TE

    def forward(self, prompt_ids, noisy_latents, timesteps):
        encoder_hidden_states = self.TE(prompt_ids, output_hidden_states=True)  # Get the text embedding for conditioning
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample  # Predict the noise residual
        return model_pred

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

def import_text_encoder_class(pretrained_model_name_or_path: str, revision: str):
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