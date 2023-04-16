import argparse
import sys
import torch
from loguru import logger

from hcpdiff.train_ac import Trainer, TextImagePairDataset, RatioBucket, load_config_with_cli
from hcpdiff.data import collate_fn_ft

class TrainerSingleCard(Trainer):
    def build_data(self, cfg_data):
        train_dataset = TextImagePairDataset(cfg_data, self.tokenizer, tokenizer_repeats=self.cfgs.model.tokenizer_repeats)
        if isinstance(train_dataset.bucket, RatioBucket):
            arb=True
            train_dataset.bucket.make_arb(cfg_data.batch_size*self.world_size)
        else:
            arb=False

        logger.info(f"len(train_dataset): {len(train_dataset)}")

        if cfg_data.cache_latents:
            self.cache_latents = True
            train_dataset.cache_latents(self.vae, self.weight_dtype, self.device)

        # Pytorch Data loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg_data.batch_size,
            num_workers=self.cfgs.train.workers, shuffle=not arb, collate_fn=collate_fn_ft)
        return train_loader, arb

    def update_ema(self):
        if hasattr(self, 'ema_unet'):
            self.ema_unet.step(self.unet.named_parameters())
        if hasattr(self, 'ema_text_encoder'):
            self.ema_text_encoder.step(self.text_encoder.named_parameters())

    def get_unet_raw(self):
        return self.unet

    def get_text_encoder_raw(self):
        return self.text_encoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, _ = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=sys.argv[3:]) # skip --cfg
    trainer=TrainerSingleCard(conf)
    trainer.train()