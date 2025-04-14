import torch
from accelerate import infer_auto_device_map, dispatch_model
from diffusers.utils.import_utils import is_xformers_available
from rainbowneko.infer import BasicAction

from hcpdiff.utils.net_utils import get_dtype
from hcpdiff.utils.net_utils import to_cpu
from hcpdiff.utils.utils import size_to_int, int_to_size

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
    def __init__(self, max_VRAM: str, max_RAM: str, vae_cpu=False, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.max_VRAM = max_VRAM
        self.max_RAM = max_RAM
        self.vae_cpu = vae_cpu

    def forward(self, vae, denoiser, dtype: str, **states):
        # denoiser offload
        torch_dtype = get_dtype(dtype)
        vram = size_to_int(self.max_VRAM)
        device_map = infer_auto_device_map(denoiser, max_memory={0:int_to_size(vram >> 1), "cpu":self.max_RAM}, dtype=torch_dtype)
        denoiser = dispatch_model(denoiser, device_map)

        device_map = infer_auto_device_map(vae, max_memory={0:int_to_size(vram >> 5), "cpu":self.max_RAM}, dtype=torch_dtype)
        vae = dispatch_model(vae, device_map)
        # VAE offload
        vram = size_to_int(self.max_VRAM)
        if not self.vae_cpu:
            device_map = infer_auto_device_map(vae, max_memory={0:int_to_size(vram >> 5), "cpu":self.max_RAM}, dtype=torch.float32)
            vae = dispatch_model(vae, device_map)
        else:
            to_cpu(vae)
            vae_decode_raw = vae.decode

            def vae_decode_offload(latents, return_dict=True, decode_raw=vae.decode):
                vae.to(dtype=torch.float32)
                res = decode_raw(latents.cpu().to(dtype=torch.float32), return_dict=return_dict)
                return res

            vae.decode = vae_decode_offload

            vae_encode_raw = vae.encode

            def vae_encode_offload(x, return_dict=True, encode_raw=vae.encode):
                vae.to(dtype=torch.float32)
                res = encode_raw(x.cpu().to(dtype=torch.float32), return_dict=return_dict)
                return res

            vae.encode = vae_encode_offload
            return {'denoiser':denoiser, 'vae':vae, 'vae_decode_raw':vae_decode_raw, 'vae_encode_raw':vae_encode_raw}

        return {'denoiser':denoiser, 'vae':vae}

class XformersEnableAction(BasicAction):
    def forward(self, denoiser, **states):
        if is_xformers_available():
            denoiser.enable_xformers_memory_efficient_attention()
            # self.te_hook.enable_xformers()
