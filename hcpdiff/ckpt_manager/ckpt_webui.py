from .base import CkptManagerBase
import os
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from hcpdiff.models.plugin import BasePluginBlock
from hcpdiff.tools.sd2diffusers import load_sd_ckpt, patch_method

class CkptManagerWebui(CkptManagerBase):

    def set_save_dir(self, save_dir, emb_dir=None):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.emb_dir = emb_dir

    def save(self, step, unet, TE, lora_unet, lora_TE, all_plugin_unet, all_plugin_TE, embs, pipe: StableDiffusionPipeline, **kwargs):
        def state_dict_unet(*args, model=unet, **kwargs):
            plugin_names = {k for k, v in model.named_modules() if isinstance(v, BasePluginBlock)}
            model_sd = {}
            for k, v in model.state_dict_().items():
                for name in plugin_names:
                    if k.startswith(name):
                        break
                else:
                    model_sd[k] = v
            return model_sd
        unet.state_dict_ = unet.state_dict
        unet.state_dict = state_dict_unet

        def state_dict_TE(*args, model=TE, **kwargs):
            plugin_names = {k for k, v in model.named_modules() if isinstance(v, BasePluginBlock)}
            model_sd = {}
            for k, v in model.state_dict_().items():
                for name in plugin_names:
                    if k.startswith(name):
                        break
                else:
                    model_sd[k] = v
            return model_sd
        TE.state_dict_ = TE.state_dict
        TE.state_dict = state_dict_TE

        pipe.save_pretrained(os.path.join(self.save_dir, f"model-{step}"), **kwargs)

    @classmethod
    def load(cls, pretrained_model, original_config_file, from_safetensors=False, device='cpu', ema=True, **kwargs) -> StableDiffusionPipeline:
        patch_method()
        pipe = load_sd_ckpt(
            checkpoint_path=pretrained_model,
            original_config_file=original_config_file,
            extract_ema=ema,
            scheduler_type='pndm',
            from_safetensors=from_safetensors,
            device=device,
        )
        return pipe