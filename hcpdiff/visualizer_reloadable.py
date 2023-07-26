from hcpdiff.visualizer import Visualizer
import accelerate.hooks
from omegaconf import OmegaConf
from hcpdiff.models import EmbeddingPTHook
import hydra
from diffusers import AutoencoderKL, PNDMScheduler
import torch
from hcpdiff.utils.cfg_net_tools import load_hcpdiff, make_plugin
from hcpdiff.utils import load_config, hash_str
from copy import deepcopy

class VisualizerReloadable(Visualizer):
    def __init__(self, cfgs):
        self.lora_dict = {}
        self.part_plugin_cfg_set = set()
        super().__init__(cfgs)

    def _merge_model(self, cfg_merge):
        if 'plugin_cfg' in cfg_merge:  # Build plugins
            plugin_cfg = hydra.utils.instantiate(load_config(cfg_merge.plugin_cfg))
            make_plugin(self.pipe.unet, plugin_cfg.plugin_unet)
            make_plugin(self.pipe.text_encoder, plugin_cfg.plugin_TE)

        for cfg_group in cfg_merge.values():
            if hasattr(cfg_group, 'type'):
                if cfg_group.type == 'unet':
                    lora_group = load_hcpdiff(self.pipe.unet, cfg_group)
                elif cfg_group.type == 'TE':
                    lora_group = load_hcpdiff(self.pipe.text_encoder, cfg_group)
                else:
                    raise ValueError(f'no host model type named {cfg_group.type}')

                # record all lora plugin with its config hash
                if not lora_group.empty():
                    for cfg_lora, lora_plugin in zip(cfg_group.lora, lora_group.plugin_dict.values()):
                        self.lora_dict[hash_str(OmegaConf.to_yaml(cfg_lora, resolve=True))] = lora_plugin

                # record all part and plugin config hash
                for cfg_part in getattr(cfg_group, "part", None) or []:
                    self.part_plugin_cfg_set.add(hash_str(OmegaConf.to_yaml(cfg_part, resolve=True)))
                for cfg_plugin in getattr(cfg_group, "plugin", None) or []:
                    self.part_plugin_cfg_set.add(hash_str(OmegaConf.to_yaml(cfg_plugin, resolve=True)))

    def merge_model(self):
        self.part_plugin_cfg_set.clear()
        self.lora_dict.clear()
        self._merge_model(self.cfg_merge)

    def part_plugin_changed(self):
        if not self.cfg_merge:
            return not self.cfg_same(self.cfg_merge, self.cfgs_old.merge)
        part_plugin_cfg_set_new = set()
        for cfg_group in self.cfg_merge.values():
            for cfg_part in getattr(cfg_group, "part", None) or []:
                part_plugin_cfg_set_new.add(hash_str(OmegaConf.to_yaml(cfg_part, resolve=True)))
            for cfg_plugin in getattr(cfg_group, "plugin", None) or []:
                part_plugin_cfg_set_new.add(hash_str(OmegaConf.to_yaml(cfg_plugin, resolve=True)))
        return part_plugin_cfg_set_new !=  self.part_plugin_cfg_set

    @staticmethod
    def cfg_same(cfg1, cfg2):
        if cfg1 is None:
            return cfg2 is None
        elif cfg2 is None:
            return cfg1 is None
        else:
            return OmegaConf.to_yaml(cfg1) == OmegaConf.to_yaml(cfg2)

    def reload_offload(self) -> bool:
        if not self.cfg_same(self.cfgs_raw.offload, self.cfgs_raw_old.offload):
            if self.offload_old:
                # remove offload hooks
                accelerate.hooks.remove_hook_from_module(self.pipe.unet, recurse=True)
                accelerate.hooks.remove_hook_from_module(self.pipe.vae, recurse=True)
        else:
            return False

        if self.offload:
            self.pipe.unet.to('cpu')
            self.pipe.vae.to('cpu')
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.build_offload(self.cfgs.offload)
        else:
            self.pipe.unet.to('cuda')
        return True

    def reload_emb_hook(self) -> bool:
        if self.cfgs.emb_dir!=self.cfgs_old.emb_dir or self.cfgs.N_repeats!=self.cfgs_old.N_repeats:
            self.emb_hook.remove()
            self.emb_hook, _ = EmbeddingPTHook.hook_from_dir(self.cfgs.emb_dir, self.pipe.tokenizer, self.pipe.text_encoder,
                                                         N_repeats=self.cfgs.N_repeats)
            return True
        return False

    def reload_te_hook(self) -> bool:
        if self.cfgs.clip_skip != self.cfgs_old.clip_skip or self.cfgs.N_repeats != self.cfgs_old.N_repeats:
            self.te_hook.N_repeats = self.cfgs.N_repeats
            self.te_hook.clip_skip = self.cfgs.clip_skip
            return True
        return False

    def reload_model(self) -> bool:
        pipeline = self.get_pipeline()
        if self.cfgs.pretrained_model!=self.cfgs_old.pretrained_model or self.part_plugin_changed():
            comp = pipeline.from_pretrained(self.cfgs.pretrained_model, safety_checker=None, requires_safety_checker=False,
                                        torch_dtype=self.dtype).components
            if 'vae' in self.cfgs.new_components:
                self.cfgs.new_components.vae = hydra.utils.instantiate(self.cfgs.new_components.vae)
            comp.update(self.cfgs.new_components)
            self.pipe = pipeline(**comp)
            if self.cfg_merge:
                self.merge_model()
            self.pipe = self.pipe.to(torch_dtype=self.dtype)
            self.build_optimize()
            return True
        return False

    def reload_pipe(self) -> bool:
        pipeline = self.get_pipeline()
        if type(self.pipe)!=pipeline:
            self.pipe = pipeline(**self.pipe.components)
            return True
        return False


    def reload_scheduler(self) -> bool:
        if 'scheduler' in self.cfgs_raw_old.new_components and 'scheduler' not in self.cfgs_raw.new_components:
            # load default scheduler
            self.pipe.scheduler = PNDMScheduler.from_pretrained(self.cfgs.pretrained_model, subfolder='scheduler', torch_dtype=self.dtype)
            return True
        elif not self.cfg_same(getattr(self.cfgs_raw_old.new_components, 'scheduler', {}), getattr(self.cfgs_raw.new_components, 'scheduler', {})):
            self.pipe.scheduler = self.cfgs.new_components.scheduler
            return True
        return False

    def reload_vae(self) -> bool:
        if 'vae' in self.cfgs_raw_old.new_components and 'vae' not in self.cfgs_raw.new_components:
            # load default VAE
            self.cfgs.new_components.vae = AutoencoderKL.from_pretrained(self.cfgs.pretrained_model, subfolder='vae', torch_dtype=self.dtype)
            return True
        elif not self.cfg_same(getattr(self.cfgs_raw_old.new_components, 'vae', {}), getattr(self.cfgs_raw.new_components, 'vae', {})):
            # VAE config changed, need reload
            if 'vae' in self.cfgs_old.new_components:
                del self.cfgs_old.new_components.vae
                torch.cuda.empty_cache()
            self.cfgs.new_components.vae = hydra.utils.instantiate(self.cfgs.new_components.vae)
            self.pipe.vae = self.cfgs.new_components.vae
            return True
        return False

    def reload_lora(self):
        if self.cfg_merge is None:
            if self.cfgs_old.merge is None:
                return False
            else:
                for lora in self.lora_dict.values():
                    lora.remove()
                self.lora_dict.clear()
                return True

        cfg_merge = deepcopy(self.cfg_merge)
        all_lora_hash = set()
        for k, cfg_group in self.cfg_merge.items():
            if 'part' in cfg_merge[k]:
                del cfg_merge[k].part
            if 'plugin' in cfg_merge[k]:
                del cfg_merge[k].plugin

            lora_add = []
            for cfg_lora in getattr(cfg_group, "lora", None) or []:
                cfg_hash = hash_str(OmegaConf.to_yaml(cfg_lora, resolve=True))
                if cfg_hash not in self.lora_dict:
                    lora_add.append(cfg_lora)
                    all_lora_hash.add(cfg_hash)
            cfg_merge[k].lora = OmegaConf.create(lora_add)

        lora_rm_set = set(self.lora_dict.keys())-all_lora_hash
        for cfg_hash in lora_rm_set:
            self.lora_dict[cfg_hash].remove()
        for cfg_hash in lora_rm_set:
            del self.lora_dict[cfg_hash]

        self._merge_model(cfg_merge)

    def check_reload(self, cfgs):
        '''
        Reload and modify each module based on the changes of configuration file.
        '''
        self.cfgs_raw_old = self.cfgs_raw
        self.cfgs_old = self.cfgs
        self.offload_old = self.offload

        self.cfgs_raw = cfgs

        # Reload vae only when vae config changes
        if 'vae' in self.cfgs_raw.new_components:
            vae_cfg = self.cfgs_raw.new_components.vae
            self.cfgs_raw.new_components.vae = None
            self.cfgs = hydra.utils.instantiate(self.cfgs_raw)
            self.cfgs_raw.new_components.vae = vae_cfg
            self.cfgs.new_components.vae = vae_cfg
        else:
            self.cfgs = hydra.utils.instantiate(self.cfgs_raw)

        self.cfg_merge = self.cfgs.merge
        self.offload = 'offload' in self.cfgs and self.cfgs.offload is not None
        self.dtype = self.dtype_dict[self.cfgs.dtype]

        self.need_inter_imgs = any(item.need_inter_imgs for item in self.cfgs.interface)

        is_model_reload = self.reload_model()
        if not is_model_reload:
            is_vae_reload = self.reload_vae()
            if is_vae_reload:
                self.build_vae_offload()
            self.reload_lora()
            self.reload_scheduler()
            self.reload_offload()
            self.reload_emb_hook()
            self.reload_te_hook()
            self.reload_pipe()

        if getattr(self.cfgs, 'vae_optimize', None) is not None:
            if self.cfgs.vae_optimize.tiling:
                self.pipe.vae.enable_tiling()
            else:
                self.pipe.vae.disable_tiling()

            if self.cfgs.vae_optimize.slicing:
                self.pipe.vae.enable_slicing()
            else:
                self.pipe.vae.disable_slicing()

        del self.cfgs_raw_old
        del self.cfgs_old

