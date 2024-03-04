from accelerate import infer_auto_device_map, dispatch_model
from diffusers.utils.import_utils import is_xformers_available

from hcpdiff.utils.net_utils import get_dtype, to_cpu, to_cuda
from hcpdiff.utils.utils import size_to_int, int_to_size
from hcpdiff.utils import load_config
from hcpdiff.utils.cfg_net_tools import make_plugin
import hydra
from .base import BasicAction, from_memory_context, MemoryMixin

class VaeOptimizeAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, vae=None, slicing=True, tiling=False):
        super().__init__()
        self.slicing = slicing
        self.tiling = tiling
        self.vae = vae

    def forward(self, memory, **states):
        vae = self.vae or memory.vae

        if self.tiling:
            vae.enable_tiling()
        if self.slicing:
            vae.enable_slicing()
        return states

class BuildOffloadAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, max_VRAM: str, max_RAM: str):
        super().__init__()
        self.max_VRAM = max_VRAM
        self.max_RAM = max_RAM

    def forward(self, memory, dtype: str, **states):
        torch_dtype = get_dtype(dtype)
        vram = size_to_int(self.max_VRAM)
        device_map = infer_auto_device_map(memory.unet, max_memory={0:int_to_size(vram >> 1), "cpu":self.max_RAM}, dtype=torch_dtype)
        memory.unet = dispatch_model(memory.unet, device_map)

        device_map = infer_auto_device_map(memory.vae, max_memory={0:int_to_size(vram >> 5), "cpu":self.max_RAM}, dtype=torch_dtype)
        memory.vae = dispatch_model(memory.vae, device_map)
        return {'dtype':dtype, **states}

class XformersEnableAction(BasicAction, MemoryMixin):
    def forward(self, memory, **states):
        if is_xformers_available():
            memory.unet.enable_xformers_memory_efficient_attention()
            # self.te_hook.enable_xformers()
        return states

class StartTextEncode(BasicAction, MemoryMixin):
    def forward(self, memory, **states):
        to_cuda(memory.text_encoder)
        return states

class EndTextEncode(BasicAction, MemoryMixin):
    def forward(self, memory, **states):
        to_cpu(memory.text_encoder)
        return states

class StartDiffusion(BasicAction, MemoryMixin):
    def forward(self, memory, **states):
        to_cuda(memory.unet)
        return states

class EndDiffusion(BasicAction, MemoryMixin):
    def forward(self, memory, **states):
        to_cpu(memory.unet)
        return states

class BuildPluginAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, model: str, cfg):
        self.model = model
        self.plugin_cfg = cfg

    def forward(self, memory, **states):
        if isinstance(self.plugin_cfg, str):
            plugin_cfg = load_config(self.plugin_cfg)
            plugin_cfg = {'plugin_unet':hydra.utils.instantiate(plugin_cfg['plugin_unet']),
                'plugin_TE':hydra.utils.instantiate(plugin_cfg['plugin_TE'])}
        else:
            plugin_cfg = self.plugin_cfg
        all_plugin_group_unet = make_plugin(memory.unet, plugin_cfg['plugin_unet'])
        all_plugin_group_TE = make_plugin(memory.text_encoder, plugin_cfg['plugin_TE'])

        if 'plugin_dict' not in memory:
            memory.plugin_dict = {}

        for name, plugin_group in all_plugin_group_unet.items():
            memory.plugin_dict[name] = plugin_group
        for name, plugin_group in all_plugin_group_TE.items():
            memory.plugin_dict[name] = plugin_group

        return states