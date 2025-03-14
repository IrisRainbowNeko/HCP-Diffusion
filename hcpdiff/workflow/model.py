from accelerate import infer_auto_device_map, dispatch_model
from diffusers.utils.import_utils import is_xformers_available

from hcpdiff.utils.net_utils import get_dtype
from hcpdiff.utils.utils import size_to_int, int_to_size
from rainbowneko.infer import BasicAction

class VaeOptimizeAction(BasicAction):
    def __init__(self, slicing=True, tiling=False, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.slicing = slicing
        self.tiling = tiling

    def forward(self, vae, **states):
        if self.tiling:
            vae.enable_tiling()
        if self.slicing:
            vae.enable_slicing()

class BuildOffloadAction(BasicAction):
    def __init__(self, max_VRAM: str, max_RAM: str, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.max_VRAM = max_VRAM
        self.max_RAM = max_RAM

    def forward(self, vae, denoiser, dtype: str, **states):
        torch_dtype = get_dtype(dtype)
        vram = size_to_int(self.max_VRAM)
        device_map = infer_auto_device_map(denoiser, max_memory={0:int_to_size(vram >> 1), "cpu":self.max_RAM}, dtype=torch_dtype)
        denoiser = dispatch_model(denoiser, device_map)

        device_map = infer_auto_device_map(vae, max_memory={0:int_to_size(vram >> 5), "cpu":self.max_RAM}, dtype=torch_dtype)
        vae = dispatch_model(vae, device_map)
        return {'denoiser':denoiser, 'vae':vae}

class XformersEnableAction(BasicAction):
    def forward(self, denoiser, **states):
        if is_xformers_available():
            denoiser.enable_xformers_memory_efficient_attention()
            # self.te_hook.enable_xformers()
