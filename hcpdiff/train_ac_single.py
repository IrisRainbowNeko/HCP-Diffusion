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

    def prepare(self):
        # Prepare everything with accelerator.
        prepare_obj_list = [self.unet]
        prepare_name_list = ['unet']
        if hasattr(self, 'optimizer'):
            prepare_obj_list.extend([self.optimizer, self.lr_scheduler])
            prepare_name_list.extend(['optimizer', 'lr_scheduler'])
        if hasattr(self, 'optimizer_pt'):
            prepare_obj_list.extend([self.optimizer_pt, self.lr_scheduler_pt])
            prepare_name_list.extend(['optimizer_pt', 'lr_scheduler_pt'])
        if self.train_TE:
            prepare_obj_list.append(self.text_encoder)
            prepare_name_list.append('text_encoder')

        prepare_obj_list.extend(self.train_loader_group.loader_list)
        prepared_obj = self.accelerator.prepare(*prepare_obj_list)

        ds_num = len(self.train_loader_group.loader_list)
        self.train_loader_group.loader_list = list(prepared_obj[-ds_num:])
        prepared_obj = prepared_obj[:-ds_num]

        for name, obj in zip(prepare_name_list, prepared_obj):
            setattr(self, name, obj)

    def build_data(self, data_builder: partial) -> torch.utils.data.DataLoader:
        train_dataset, batch_size, arb = self.build_dataset(data_builder)

        # Pytorch Data loader
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=self.world_size,
                                                                        rank=self.local_rank, shuffle=not arb)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=self.cfgs.train.workers,
                                                   sampler=train_sampler, collate_fn=train_dataset.collate_fn)
        return train_loader

    def encode_decode(self, prompt_ids, noisy_latents, timesteps, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps, **kwargs)
        if hasattr(self.text_encoder, 'input_feeder'):
            for feeder in self.text_encoder.input_feeder:
                feeder(input_all)
        if hasattr(self.unet_raw, 'input_feeder'):
            for feeder in self.unet_raw.input_feeder:
                feeder(input_all)

        encoder_hidden_states = self.text_encoder(prompt_ids, output_hidden_states=True)  # Get the text embedding for conditioning
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample  # Predict the noise residual
        return model_pred

    def get_loss(self, model_pred, target, timesteps, att_mask):
        return (self.criterion(model_pred.float(), target.float(), timesteps)*att_mask).mean()

    def update_ema(self):
        if hasattr(self, 'ema_unet'):
            self.ema_unet.step(self.unet.named_parameters())
        if hasattr(self, 'ema_text_encoder'):
            self.ema_text_encoder.step(self.text_encoder.named_parameters())

    @property
    def unet_raw(self):
        return self.unet

    @property
    def text_encoder_raw(self):
        return self.text_encoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, _ = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=sys.argv[3:])  # skip --cfg
    trainer = TrainerSingleCard(conf)
    trainer.train()
