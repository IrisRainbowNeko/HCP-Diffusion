"""
train_colo.py
====================
    :Name:        train with colossal-AI
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     MIT
"""

import argparse
import os
import sys
import torch
from torch import nn

import colossalai
import colossalai.tensor
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.parallel.utils import get_static_torch_model
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext

from train_ac import Trainer, get_scheduler, ModelEMA
from diffusers import UNet2DConditionModel
from data.utils import collate_fn_ft
from utils.colo_utils import gemini_zero_dpp, GeminiAdamOptimizerP
from utils.utils import import_model_class_from_model_name_or_path, load_config_with_cli

class TEUnetWapper(nn.Module):
    def __init__(self, unet, TE):
        super().__init__()
        self.unet = unet
        self.TE = TE

    def forward(self, prompt_ids, noisy_latents, timesteps):
        encoder_hidden_states = self.TE(prompt_ids)  # Get the text embedding for conditioning
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample  # Predict the noise residual
        return model_pred

class TrainerColo(Trainer):
    def init_context(self, cfgs_raw):
        # If passed along, set the training seed now.
        if self.cfgs.seed is None:
            colossalai.launch_from_torch(config='./config.py')
        else:
            # set_seed(args.seed)
            colossalai.launch_from_torch(config='./config.py', seed=self.cfgs.seed)

        self.local_rank = gpc.get_local_rank(ParallelMode.DATA)
        self.world_size = gpc.get_world_size(ParallelMode.DATA)

    @property
    def device(self):
        return get_current_device()

    @property
    def is_local_main_process(self):
        return self.local_rank in [0,-1]

    def prepare(self):
        pass

    def wait_for_everyone(self):
        torch.cuda.synchronize()

    def build_unet_and_TE(self):
        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(self.cfgs.model.pretrained_model_name_or_path,
                                                                      self.cfgs.model.revision)
        with ColoInitContext(device=self.device):
            self.unet = UNet2DConditionModel.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfgs.model.revision,
                low_cpu_mem_usage=False
            )
            if self.train_TE:
                self.text_encoder = text_encoder_cls.from_pretrained(
                    self.cfgs.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfgs.model.revision
                )
        if not self.train_TE:
            self.text_encoder = text_encoder_cls.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfgs.model.revision
            )

    def build_ema(self):
        if self.cfgs.model.ema_unet>0:
            self.ema_unet = ModelEMA(get_static_torch_model(self.unet).named_parameters(), self.cfgs.model.ema_unet)
        if self.train_TE and self.cfgs.model.ema_text_encoder>0:
            self.ema_text_encoder = ModelEMA(self.text_encoder.named_parameters(), self.cfgs.model.ema_text_encoder)

    def get_param_group_train(self):
        with ColoInitContext(device=self.device):
            params = super().get_param_group_train()

        self.lora_unet.set_inplace(False)
        if self.DA_lora:
            self.lora_unet_neg.set_inplace(False)
        if self.train_TE:
            self.lora_TE.set_inplace(False)
            if self.DA_lora:
                self.lora_TE_neg.set_inplace(False)
            self.TE_unet = gemini_zero_dpp(TEUnetWapper(self.unet, self.text_encoder))
        else:
            self.unet = gemini_zero_dpp(self.unet)
        return params

    def build_optimizer_scheduler(self):
        # set optimizer
        parameters, parameters_pt = self.get_param_group_train()

        cfg_opt = self.cfgs.train.optimizer
        if len(parameters)>0: # do fine-tuning
            if self.cfgs.train.scale_lr:
                self.scale_lr(parameters)
            self.optimizer = GeminiAdamOptimizerP(self.TE_unet if self.train_TE else self.unet, parameters, lr=self.lr,
                                                  initial_scale=2 ** 5, clipping_norm=self.cfgs.train.max_grad_norm)

            self.lr_scheduler = get_scheduler(optimizer=self.optimizer, **self.cfgs.train.scheduler)

        if len(parameters_pt)>0: # do prompt-tuning
            if self.cfgs.train.scale_lr_pt:
                self.scale_lr(parameters_pt)

            self.optimizer_pt = torch.optim.AdamW(params=parameters_pt, lr=self.lr, weight_decay=cfg_opt.weight_decay_pt)
            self.lr_scheduler_pt = get_scheduler(optimizer=self.optimizer_pt, **self.cfgs.train.scheduler_pt)

    def encode_decode(self, prompt_ids, noisy_latents, timesteps):
        if self.train_TE:
            model_pred = self.TE_unet(prompt_ids, noisy_latents, timesteps)
        else:
            model_pred = super(TrainerColo, self).encode_decode(prompt_ids, noisy_latents, timesteps)
        return model_pred

    def train_one_step(self, image, att_mask, prompt_ids):
        torch.cuda.reset_peak_memory_stats()
        image = image.to(self.device, dtype=self.weight_dtype, non_blocking=True)
        att_mask = att_mask.to(self.device, non_blocking=True)
        prompt_ids = prompt_ids.to(self.device, non_blocking=True)

        latents = self.get_latents(image, self.train_loader.dataset)
        model_pred, target = self.forward(latents, prompt_ids)

        if self.train_loader_class is not None:
            # DreamBooth prior forward
            image_cls, att_mask_cls, prompt_ids_cls = next(self.data_iter_class)
            image_cls = image_cls.to(self.device, dtype=self.weight_dtype)
            att_mask_cls = att_mask_cls.to(self.device)
            prompt_ids_cls = prompt_ids_cls.to(self.device)
            latents_cls = self.get_latents(image_cls, self.train_loader_class.dataset)
            model_pred_prior, target_prior = self.forward(latents_cls, prompt_ids_cls)

            loss = self.get_loss(model_pred, target, att_mask)  # Compute instance loss
            prior_loss = self.get_loss(model_pred_prior, target_prior, att_mask_cls)  # Compute prior loss
            loss = loss + self.cfgs.train.loss.prior_loss_weight * prior_loss
        else:
            loss = self.get_loss(model_pred, target, att_mask)

        if hasattr(self, 'optimizer'):
            self.optimizer.backward(loss)
        else:
            loss.backward()

        if hasattr(self, 'optimizer'):
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=self.cfgs.train.set_grads_to_none)

        if hasattr(self, 'optimizer_pt'): # prompt tuning
            self.optimizer_pt.step()
            self.lr_scheduler_pt.step()
            self.optimizer_pt.zero_grad(set_to_none=self.cfgs.train.set_grads_to_none)

        self.update_ema()
        return loss.item()

    def update_ema(self):
        if hasattr(self, 'ema_unet'):
            self.ema_unet.step(get_static_torch_model(self.unet).named_parameters())
        if hasattr(self, 'ema_text_encoder'):
            self.ema_text_encoder.step(self.text_encoder.named_parameters())

    def get_unet_raw(self):
        if self.train_TE:
            unet = get_static_torch_model(self.TE_unet).unet
        else:
            unet = get_static_torch_model(self.unet)
        req_grad_dict = {k:v.requires_grad for k,v in self.unet.named_parameters()}
        for k,v in unet.named_parameters():
            v.requires_grad = req_grad_dict[k]
        return unet

    def get_text_encoder_raw(self):
        if self.train_TE:
            TE = get_static_torch_model(self.TE_unet).TE
            req_grad_dict = {k: v.requires_grad for k, v in self.text_encoder.named_parameters()}
            for k, v in TE.named_parameters():
                v.requires_grad = req_grad_dict[k]
        else:
            TE = self.text_encoder
        return TE

    def save_model(self, from_raw=True):
        super(TrainerColo, self).save_model(from_raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    parser.add_argument( "--placement", type=str, default="cpu",
        help="Placement Policy for Gemini. Valid when using colossalai as dist plan.")
    args, _ = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=sys.argv[3:]) # skip --cfg
    trainer=TrainerColo(conf)
    trainer.train()