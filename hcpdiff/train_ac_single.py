import argparse
import sys
from functools import partial

import torch
from accelerate import Accelerator
from loguru import logger

from hcpdiff.train_ac import Trainer, RatioBucket, load_config_with_cli, set_seed

class TrainerSingleCard(Trainer):
    def init_context(self, cfgs_raw):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfgs.train.gradient_accumulation_steps,
            mixed_precision=self.cfgs.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        self.local_rank = 0
        self.world_size = self.accelerator.num_processes

        set_seed(self.cfgs.seed+self.local_rank)

    def build_data(self, data_builder: partial) -> torch.utils.data.DataLoader:
        train_dataset, batch_size, arb = self.build_dataset(data_builder)

        # Pytorch Data loader
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=self.world_size,
                                                                        rank=self.local_rank, shuffle=not arb)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=self.cfgs.train.workers,
                                                   sampler=train_sampler, collate_fn=train_dataset.collate_fn)
        return train_loader

    def get_loss(self, model_pred, target, timesteps, att_mask):
        return (self.criterion(model_pred.float(), target.float(), timesteps)*att_mask).mean()

    def update_ema(self):
        if hasattr(self, 'ema_unet'):
            self.ema_unet.step(self.unet_raw.named_parameters())
        if hasattr(self, 'ema_text_encoder'):
            self.ema_text_encoder.step(self.TE_raw.named_parameters())

    @property
    def unet_raw(self):
        return self.TE_unet.unet

    @property
    def TE_raw(self):
        return self.TE_unet.TE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerSingleCard(conf)
    trainer.train()
