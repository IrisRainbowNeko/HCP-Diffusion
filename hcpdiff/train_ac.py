import argparse
import warnings

import torch
from rainbowneko.parser import load_config_with_cli
from rainbowneko.train import Trainer
from rainbowneko.utils import xformers_available, is_dict
from hcpdiff.ckpt_manager import EmbFormat

class HCPTrainer(Trainer):
    def config_model(self):
        if self.cfgs.model.enable_xformers:
            if xformers_available:
                self.model_wrapper.enable_xformers()
            else:
                warnings.warn("xformers is not available. Make sure it is installed correctly")

        self.model_wrapper.requires_grad_(False)
        self.model_wrapper.eval()
        self.weight_dtype = self.weight_dtype_map.get(self.cfgs.mixed_precision, torch.float32)
        self.vae_dtype = self.weight_dtype_map.get(self.cfgs.model.get('vae_dtype', None), torch.float32)
        self.model_wrapper.set_dtype(self.weight_dtype, self.vae_dtype)

        if self.cfgs.model.gradient_checkpointing:
            self.model_wrapper.enable_gradient_checkpointing()

    def get_param_group_train(self):
        train_params = super().get_param_group_train()

        # For prompt-tuning
        if self.cfgs.emb_pt is None:
            train_params_emb, self.train_pts = [], {}
        else:
            train_params_emb, self.train_pts = self.cfgs.emb_pt.get_params_group(self.model_wrapper)
            self.emb_format = EmbFormat()
        train_params += train_params_emb
        return train_params

    def get_loss(self, ds_name, model_pred, inputs):
        loss = super().get_loss(ds_name, model_pred, inputs)
        # make DDP happy
        if len(self.train_pts)>0:
            loss = loss+0*sum([emb.mean() for emb in self.train_pts.values()])
        return loss

    def save_model(self, from_raw=False):
        for manager in self.ckpt_manager:
            manager.save_step(
                self.model_raw,
                name=self.cfgs.model.name,
                step=self.real_step,
                prefix=self.ckpt_dir,
                model_ema=getattr(self, "ema_model", None),
            )
            try:
                manager.save_plugins(
                    self.model_raw,
                    self.all_plugin,
                    name=self.cfgs.model.name,
                    step=self.real_step,
                    model_ema=getattr(self, "ema_model", None),
                )
            except:
                self.loggers.info(f"{manager} not support to save plugin!")

                import traceback
                traceback.print_exc()

            try:
                for pt_name, pt in self.train_pts:
                    manager.source.put(pt_name, pt, self.emb_format, prefix=self.ckpt_dir)
            except:
                self.loggers.info(f"{manager} not support to save embedding!")

        self.loggers.info(f"Saved state, step: {self.real_step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCP Diffusion Trainer")
    parser.add_argument("--cfg", type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = HCPTrainer(parser, conf)
    trainer.train()
