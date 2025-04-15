import argparse
import warnings

import torch
from rainbowneko.ckpt_manager import NekoPluginSaver
from rainbowneko.train.trainer import TrainerDeepspeed
from rainbowneko.utils import xformers_available

from hcpdiff.trainer_ac import HCPTrainer, load_config_with_cli

class HCPTrainerDeepspeed(TrainerDeepspeed, HCPTrainer):
    def config_model(self):
        if self.cfgs.model.enable_xformers:
            if xformers_available:
                self.model_wrapper.enable_xformers()
            else:
                warnings.warn("xformers is not available. Make sure it is installed correctly")

        if self.model_wrapper.vae is not None:
            self.vae_dtype = self.weight_dtype_map.get(self.cfgs.model.get('vae_dtype', None), torch.float32)
            self.model_wrapper.set_dtype(self.weight_dtype, self.vae_dtype)

        if self.cfgs.model.gradient_checkpointing:
            self.model_wrapper.enable_gradient_checkpointing()

        if self.is_local_main_process:
            for saver in self.ckpt_saver.values():
                if isinstance(saver, NekoPluginSaver):
                    saver.plugin_from_raw = True

def hcp_train():
    import subprocess
    parser = argparse.ArgumentParser(description='HCP-Diffusion Launcher')
    parser.add_argument('--launch_cfg', type=str, default='cfgs/launcher/deepspeed.yaml')
    args, train_args = parser.parse_known_args()

    subprocess.run(["accelerate", "launch", '--config_file', args.launch_cfg, "-m",
                       "hcpdiff.trainer_deepspeed"]+train_args, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HCP Diffusion Trainer for DeepSpeed')
    parser.add_argument("--cfg", type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = HCPTrainerDeepspeed(parser, conf)
    trainer.train()
