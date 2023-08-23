import argparse
import os
import sys
import warnings
from functools import partial

import torch

from hcpdiff.ckpt_manager import CkptManagerPKL, CkptManagerSafe
from hcpdiff.train_ac import Trainer, load_config_with_cli
from hcpdiff.utils.net_utils import get_scheduler

class TrainerDeepSpeed(Trainer):

    def build_ckpt_manager(self):
        if self.cfgs.ckpt_type == 'torch':
            self.ckpt_manager = CkptManagerPKL(plugin_from_raw=True)
        elif self.cfgs.ckpt_type == 'safetensors':
            self.ckpt_manager = CkptManagerSafe(plugin_from_raw=True)
        else:
            raise NotImplementedError(f'Not support ckpt type: {self.cfgs.ckpt_type}')
        if self.is_local_main_process:
            self.ckpt_manager.set_save_dir(os.path.join(self.exp_dir, 'ckpts'), emb_dir=self.cfgs.tokenizer_pt.emb_dir)

    @property
    def unet_raw(self):
        return self.accelerator.unwrap_model(self.TE_unet).unet if self.train_TE else self.accelerator.unwrap_model(self.TE_unet.unet)

    @property
    def TE_raw(self):
        return self.accelerator.unwrap_model(self.TE_unet).TE if self.train_TE else self.TE_unet.TE

    def build_optimizer_scheduler(self):
        # set optimizer
        parameters, parameters_pt = self.get_param_group_train()

        if len(parameters_pt)>0:  # do prompt-tuning
            cfg_opt_pt = self.cfgs.train.optimizer_pt
            if self.cfgs.train.scale_lr_pt:
                self.scale_lr(parameters_pt)
            weight_decay = getattr(cfg_opt_pt, 'weight_decay', None)
            if isinstance(cfg_opt_pt, partial):
                weight_decay = getattr(cfg_opt_pt.keywords, 'weight_decay', None)
            if weight_decay is not None:
                for param in parameters_pt:
                    param['weight_decay'] = weight_decay

            parameters += parameters_pt
            warnings.warn('deepspeed dose not support multi optimizer and lr_scheduler. optimizer_pt and scheduler_pt will not work.')

        if len(parameters)>0:
            cfg_opt = self.cfgs.train.optimizer
            if self.cfgs.train.scale_lr:
                self.scale_lr(parameters)

            if isinstance(cfg_opt, partial):
                if 'type' in cfg_opt.keywords:
                    del cfg_opt.keywords['type']
                self.optimizer = cfg_opt(params=parameters, lr=self.lr)
            elif cfg_opt.type == 'adamw_8bit':
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(params=parameters, lr=self.lr, weight_decay=cfg_opt.weight_decay)
            elif cfg_opt.type == 'deepspeed' and self.accelerator.state.deepspeed_plugin is not None:
                from deepspeed.ops.adam import FusedAdam
                self.optimizer = FusedAdam(params=parameters, lr=self.lr, weight_decay=cfg_opt.weight_decay)
            elif cfg_opt.type == 'adamw':
                self.optimizer = torch.optim.AdamW(params=parameters, lr=self.lr, weight_decay=cfg_opt.weight_decay)
            else:
                raise NotImplementedError(f'Unknown optimizer {cfg_opt.type}')

            if isinstance(self.cfgs.train.scheduler, partial):
                self.lr_scheduler = self.cfgs.train.scheduler(optimizer=self.optimizer)
            else:
                self.lr_scheduler = get_scheduler(optimizer=self.optimizer, **self.cfgs.train.scheduler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerDeepSpeed(conf)
    trainer.train()
