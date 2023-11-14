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
        self.ckpt_manager = self.ckpt_manager_map[self.cfgs.ckpt_type](plugin_from_raw=True)
        if self.is_local_main_process:
            self.ckpt_manager.set_save_dir(os.path.join(self.exp_dir, 'ckpts'), emb_dir=self.cfgs.tokenizer_pt.emb_dir)

    @property
    def unet_raw(self):
        return self.accelerator.unwrap_model(self.TE_unet).unet if self.train_TE else self.accelerator.unwrap_model(self.TE_unet.unet)

    @property
    def TE_raw(self):
        return self.accelerator.unwrap_model(self.TE_unet).TE if self.train_TE else self.TE_unet.TE

    def get_loss(self, model_pred, target, timesteps, att_mask):
        if att_mask is None:
            att_mask = 1.0
        if getattr(self.criterion, 'need_timesteps', False):
            loss = (self.criterion(model_pred.float(), target.float(), timesteps)*att_mask).mean()
        else:
            loss = (self.criterion(model_pred.float(), target.float())*att_mask).mean()
        return loss

    def build_optimizer_scheduler(self):
        # set optimizer
        parameters, parameters_pt = self.get_param_group_train()

        if len(parameters_pt)>0:  # do prompt-tuning
            cfg_opt_pt = self.cfgs.train.optimizer_pt
            # if self.cfgs.train.scale_lr_pt:
            #     self.scale_lr(parameters_pt)
            assert isinstance(cfg_opt_pt, partial), f'optimizer.type is not supported anymore, please use class path like "torch.optim.AdamW".'
            weight_decay = cfg_opt_pt.keywords.get('weight_decay', None)
            if weight_decay is not None:
                for param in parameters_pt:
                    param['weight_decay'] = weight_decay

            parameters += parameters_pt
            warnings.warn('deepspeed dose not support multi optimizer and lr_scheduler. optimizer_pt and scheduler_pt will not work.')

        if len(parameters)>0:
            cfg_opt = self.cfgs.train.optimizer
            if self.cfgs.train.scale_lr:
                self.scale_lr(parameters)
            assert isinstance(cfg_opt, partial), f'optimizer.type is not supported anymore, please use class path like "torch.optim.AdamW".'
            self.optimizer = cfg_opt(params=parameters)
            self.lr_scheduler = get_scheduler(self.cfgs.train.scheduler, self.optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerDeepSpeed(conf)
    trainer.train()
