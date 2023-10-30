"""
train_colo.py
====================
    :Name:        train with colossal-AI
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import argparse
import sys
import torch
from torch import nn

import colossalai
import colossalai.tensor
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.parallel.utils import get_static_torch_model
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils.model.colo_init_context import _convert_to_coloparam
from colossalai.tensor import ColoParameter

from hcpdiff.train_ac import Trainer, get_scheduler, ModelEMA
from diffusers import UNet2DConditionModel
from hcpdiff.utils.colo_utils import gemini_zero_dpp, GeminiAdamOptimizerP
from hcpdiff.utils.utils import load_config_with_cli
from hcpdiff.utils.net_utils import auto_text_encoder, TEUnetWrapper

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
        text_encoder_cls = auto_text_encoder(self.cfgs.model.pretrained_model_name_or_path,
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
                self.TE_unet = TEUnetWrapper(self.unet, self.text_encoder)
        if not self.train_TE:
            self.text_encoder = text_encoder_cls.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfgs.model.revision
            )

    def build_ema(self):
        if self.cfgs.model.ema_unet>0:
            self.ema_unet = ModelEMA(get_static_torch_model(self.unet).named_parameters(), self.cfgs.model.ema_unet)
        if self.train_TE and self.cfgs.model.ema_text_encoder>0:
            self.ema_text_encoder = ModelEMA(self.text_encoder.named_parameters(), self.cfgs.model.ema_text_encoder)

    def convert_emb_param(self, cinit):
        name_list = []
        for name, param in self.embedding_hook.named_parameters():
            if type(param) is ColoParameter:
                continue

            split = name.rfind('.')
            if split >= 0:  # param in submodule
                module_name = name[:split]
                param_name = name[split + 1:]
            else:
                module_name = ''  # param in current module
                param_name = name
            name_list.append((module_name, param_name))

        replaced_tensors = dict(
        )  # record mapping between (torch.Tensor, ColoTensor) to distinguish the same reference
        for module_name, param_name in name_list:
            submodule = self.embedding_hook.get_submodule(module_name)
            param = submodule.get_parameter(param_name)
            if param in replaced_tensors:
                colo_param = replaced_tensors[param]
            else:
                colo_param = _convert_to_coloparam(param, cinit._device, cinit._dtype, cinit._default_pg,
                                                   cinit._default_dist_spec)
                replaced_tensors[param] = colo_param
            delattr(submodule, param_name)
            setattr(submodule, param_name, colo_param)
            colo_param.shared_param_modules.append(submodule)

    def get_param_group_train(self):
        with ColoInitContext(device=self.device) as cinit:
            params = super().get_param_group_train()
            self.convert_emb_param(cinit)

        self.lora_unet.set_inplace(False)
        if self.DA_lora:
            self.lora_unet_neg.set_inplace(False)
        if self.train_TE:
            self.lora_TE.set_inplace(False)
            if self.DA_lora:
                self.lora_TE_neg.set_inplace(False)
            self.TE_unet = gemini_zero_dpp(self.TE_unet)
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

    def train_one_step(self, data_list):
        torch.cuda.reset_peak_memory_stats()
        loss = []
        for idx, data in enumerate(data_list):
            image = data.pop('img').to(self.device, dtype=self.weight_dtype, non_blocking=True)
            att_mask = data.pop('mask').to(self.device, non_blocking=True)
            prompt_ids = data.pop('prompt').to(self.device, non_blocking=True)
            other_datas = {k:v.to(self.device, dtype=self.weight_dtype, non_blocking=True) for k, v in data.items()}

            latents = self.get_latents(image, self.train_loader_group.get_dataset(idx))
            model_pred, target, timesteps = self.forward(latents, prompt_ids, **other_datas)
            loss.append(self.get_loss(model_pred, target, timesteps, att_mask)*self.train_loader_group.get_loss_weights(idx))
        loss = sum(loss)

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

    @property
    def unet_raw(self):
        if self.train_TE:
            unet = get_static_torch_model(self.TE_unet).unet
        else:
            unet = get_static_torch_model(self.unet)
        req_grad_dict = {k:v.requires_grad for k,v in self.unet.named_parameters()}
        for k,v in unet.named_parameters():
            v.requires_grad = req_grad_dict[k]
        return unet

    @property
    def TE_raw(self):
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
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args) # skip --cfg
    trainer=TrainerColo(conf)
    trainer.train()