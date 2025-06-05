import argparse
import warnings

import torch
from rainbowneko.parser import load_config_with_cli
from rainbowneko.ckpt_manager import NekoSaver
from rainbowneko.train.trainer import Trainer
from rainbowneko.utils import xformers_available, is_dict, weight_dtype_map
from hcpdiff.ckpt_manager import EmbFormat

class HCPTrainer(Trainer):
    def config_model(self):
        if self.cfgs.model.enable_xformers:
            if xformers_available:
                self.model_wrapper.enable_xformers()
            else:
                warnings.warn("xformers is not available. Make sure it is installed correctly")

        if self.model_wrapper.vae is not None:
            self.vae_dtype = weight_dtype_map.get(self.cfgs.model.get('vae_dtype', None), torch.float32)
            self.model_wrapper.set_dtype(self.weight_dtype, self.vae_dtype)

        if self.cfgs.model.gradient_checkpointing:
            self.model_wrapper.enable_gradient_checkpointing()

    def get_param_group_train(self):
        train_params = super().get_param_group_train()

        # For prompt-tuning
        if self.cfgs.emb_pt is None:
            train_params_emb, self.train_pts = [], {}
        else:
            from hcpdiff.parser import CfgEmbPTParser
            self.cfgs.emb_pt: CfgEmbPTParser

            train_params_emb, self.train_pts = self.cfgs.emb_pt.get_params_group(self.model_wrapper)
            self.emb_format = EmbFormat()
        train_params += train_params_emb
        return train_params

    @property
    def pt_trainable(self):
        return self.cfgs.emb_pt is not None

    def save_model(self, from_raw=False):
        NekoSaver.save_all(
            cfg=self.ckpt_saver,
            model=self.model_raw,
            plugin_groups=self.all_plugin,
            embs=self.train_pts,
            model_ema=getattr(self, "ema_model", None),
            optimizer=self.optimizer,
            name_template=f'{{}}-{self.real_step}',
        )

        self.loggers.info(f"Saved state, step: {self.real_step}")

def hcp_train():
    import subprocess
    parser = argparse.ArgumentParser(description='HCP-Diffusion Launcher')
    parser.add_argument('--launch_cfg', type=str, default='cfgs/launcher/multi.yaml')
    args, train_args = parser.parse_known_args()

    subprocess.run(["accelerate", "launch", '--config_file', args.launch_cfg, "-m",
                    "hcpdiff.trainer_ac"] + train_args, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCP Diffusion Trainer")
    parser.add_argument("--cfg", type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = HCPTrainer(parser, conf)
    trainer.train()
