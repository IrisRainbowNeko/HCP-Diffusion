from diffusers import StableDiffusionPipeline
from diffusers.models.lora import LoRACompatibleLinear

class CkptManagerBase:
    def __init__(self, **kwargs):
        pass

    def set_save_dir(self, save_dir, emb_dir=None):
        raise NotImplementedError()

    def save(self, step, unet, TE, lora_unet, lora_TE, all_plugin_unet, all_plugin_TE, embs, pipe: StableDiffusionPipeline, **kwargs):
        raise NotImplementedError()

    @classmethod
    def load(cls, pretrained_model, **kwargs) -> StableDiffusionPipeline:
        raise NotImplementedError
