import os
from typing import List
import warnings

from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler

from hcpdiff.utils import auto_text_encoder, auto_tokenizer, to_validate_file
from hcpdiff.utils.cfg_net_tools import HCPModelLoader, make_plugin
from hcpdiff.utils.img_size_tool import types_support
from hcpdiff.utils.net_utils import get_dtype
from .base import BasicAction, from_memory_context, MemoryMixin

class LoadModelsAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, pretrained_model: str, dtype: str, unet=None, text_encoder=None, tokenizer=None, vae=None, scheduler=None):
        self.pretrained_model = pretrained_model
        self.dtype = get_dtype(dtype)

        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.scheduler = scheduler

    def forward(self, memory, **states):
        memory.unet = self.unet or UNet2DConditionModel.from_pretrained(self.pretrained_model, subfolder="unet", torch_dtype=self.dtype)
        memory.text_encoder = self.text_encoder or auto_text_encoder(self.pretrained_model, subfolder="text_encoder", torch_dtype=self.dtype)
        memory.tokenizer = self.tokenizer or auto_tokenizer(self.pretrained_model, subfolder="tokenizer", use_fast=False)
        memory.vae = self.vae or AutoencoderKL.from_pretrained(self.pretrained_model, subfolder="vae", torch_dtype=self.dtype)
        memory.scheduler = self.scheduler or PNDMScheduler.from_pretrained(self.pretrained_model, subfolder="scheduler", torch_dtype=self.dtype)

        return states

class SaveImageAction(BasicAction):
    @from_memory_context
    def __init__(self, save_root: str, image_type: str = 'png', quality: int = 95):
        self.save_root = save_root
        self.image_type = image_type
        self.quality = quality

        os.makedirs(save_root, exist_ok=True)

    def forward(self, images, prompt, negative_prompt, seeds=None, **states):
        num_img_exist = max([0]+[int(x.split('-', 1)[0]) for x in os.listdir(self.save_root) if x.rsplit('.', 1)[-1] in types_support])+1

        for bid, (p, pn, img) in enumerate(zip(prompt, negative_prompt, images)):
            img_path = os.path.join(self.save_root, f"{num_img_exist}-{seeds[bid]}-{to_validate_file(prompt[0])}.{self.image_type}")
            img.save(img_path, quality=self.quality)
            num_img_exist += 1

        return {**states, 'images':images, 'prompt':prompt, 'negative_prompt':negative_prompt, 'seeds':seeds}

class BuildModelLoaderAction(BasicAction, MemoryMixin):
    def forward(self, memory, **states):
        memory.model_loader_unet = HCPModelLoader(memory.unet)
        memory.model_loader_TE = HCPModelLoader(memory.text_encoder)
        return states

class LoadPartAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, model: str, cfg):
        self.model = model
        self.cfg = cfg

    def forward(self, memory, **states):
        model_loader = memory[f"model_loader_{self.model}"]
        model_loader.load_part(self.cfg)
        return states

class LoadLoraAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, model: str, cfg):
        self.model = model
        self.cfg = cfg

    def forward(self, memory, **states):
        model_loader = memory[f"model_loader_{self.model}"]
        lora_group = model_loader.load_lora(self.cfg)
        if 'lora_dict' not in memory:
            memory.lora_dict = {}
        if path in memory.lora_dict:
            warnings.warn(f"Lora {path} already loaded, and will be replaced!")
            memory.lora_dict[path].remove()
        memory.lora_dict[path] = lora_group
        return states

class BuildPluginAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, model: str, cfg):
        self.model = model
        self.cfg = cfg

    def forward(self, memory, **states):
        if isinstance(self.cfg_merge.plugin_cfg, str):
            plugin_cfg = load_config(self.cfg_merge.plugin_cfg)
            plugin_cfg = {'plugin_unet':hydra.utils.instantiate(plugin_cfg['plugin_unet']),
                'plugin_TE':hydra.utils.instantiate(plugin_cfg['plugin_TE'])}
        else:
            plugin_cfg = self.cfg_merge.plugin_cfg
        all_plugin_group_unet = make_plugin(memory.unet, plugin_cfg['plugin_unet'])
        all_plugin_group_TE = make_plugin(memory.text_encoder, plugin_cfg['plugin_TE'])

        if 'plugin_dict' not in memory:
            memory.plugin_dict = {}

        for name, plugin_group in all_plugin_group_unet.items():
            memory.plugin_dict[name] = plugin_group
        for name, plugin_group in all_plugin_group_TE.items():
            memory.plugin_dict[name] = plugin_group

        return states

class LoadPluginAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, model: str, cfg):
        self.model = model
        self.cfg = cfg

    def forward(self, memory, **states):
        model_loader = memory[f"model_loader_{self.model}"]
        model_loader.load_plugin(self.cfg)
        return states

class RemoveLoraAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, path_list: List[str]):
        self.path_list = path_list

    def forward(self, memory, **states):
        for path in self.path_list:
            if path in memory.lora_dict:
                memory.lora_dict[path].remove()
                del memory.lora_dict[path]
            else:
                warnings.warn(f"Lora {path} not loaded!")
        return states

class RemovePluginAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, name_list: List[str]):
        self.name_list = name_list

    def forward(self, memory, **states):
        for name in self.name_list:
            if name in memory.plugin_dict:
                memory.plugin_dict[name].remove()
                del memory.plugin_dict[name]
            else:
                warnings.warn(f"Plugin {name} not loaded!")
        return states