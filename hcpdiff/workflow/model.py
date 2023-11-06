from .base import BasicAction, from_memory_context, MemoryMixin
from hcpdiff.utils.utils import load_config_with_cli, load_config, size_to_int, int_to_size, prepare_seed
from accelerate import infer_auto_device_map, dispatch_model

class VaeOptimizeAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, slicing=True, tiling=False):
        super().__init__()
        self.slicing = slicing
        self.tiling = tiling

    def forward(self, memory, **states):
        if self.tiling:
            memory.vae.enable_tiling()
        if self.slicing:
            memory.vae.enable_slicing()
        return states

class BuildOffloadAction(BasicAction, MemoryMixin):
    @from_memory_context
    def __init__(self, max_VRAM:str, max_RAM:str):
        super().__init__()
        self.max_VRAM = max_VRAM
        self.max_RAM = max_RAM

    def forward(self, memory, dtype, **states):
        vram = size_to_int(self.max_VRAM)
        device_map = infer_auto_device_map(memory.unet, max_memory={0:int_to_size(vram >> 1), "cpu":self.max_RAM}, dtype=dtype)
        memory.unet = dispatch_model(memory.unet, device_map)

        device_map = infer_auto_device_map(memory.vae, max_memory={0:int_to_size(vram >> 5), "cpu":self.max_RAM}, dtype=dtype)
        memory.vae = dispatch_model(memory.vae, device_map)
        return {'dtype':dtype, **states}