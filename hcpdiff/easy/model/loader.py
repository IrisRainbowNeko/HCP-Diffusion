import torch
from hcpdiff.ckpt_manager import DiffusersSD15Format
from hcpdiff.parser import lora_resolver
from rainbowneko.ckpt_manager import ModelManager, LocalCkptSource, CkptManagerBase
from rainbowneko.parser.model import NekoPluginLoader

class HCPLoraLoader(NekoPluginLoader):
    def __init__(self, path: str, ckpt_manager: CkptManagerBase = None, layers='all', module_to_load='', state_prefix=None,
                 base_model_alpha=0.0, load_ema=False, **plugin_kwargs):
        super().__init__(path, ckpt_manager=ckpt_manager, layers=layers, module_to_load=module_to_load, state_prefix=state_prefix,
                         resolver=lora_resolver, base_model_alpha=base_model_alpha, load_ema=load_ema, **plugin_kwargs)

def sd15_auto_loader(ckpt_path, unet=None, TE=None, vae=None, noise_sampler=None,
                     tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
    manager = ModelManager(
        format=DiffusersSD15Format(),
        source=LocalCkptSource(),
    )
    models = manager.load(ckpt_path, unet=unet, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer, revision=revision,
                          dtype=dtype, **kwargs)
    return models
