"""
train_ac.py
====================
    :Name:        train with accelerate
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import argparse
import math
import os
import time
import warnings
from functools import partial

import diffusers
import hydra
import torch
import torch.utils.checkpoint
# fix checkpoint bug for train part of model
import torch.utils.checkpoint
import torch.utils.data
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf

from hcpdiff.ckpt_manager import CkptManagerPKL, CkptManagerSafe
from hcpdiff.data import RatioBucket, DataGroup, get_sampler
from hcpdiff.deprecated.cfg_converter import TrainCFGConverter
from hcpdiff.loggers import LoggerGroup
from hcpdiff.models import CFGContext, DreamArtistPTContext, TEUnetWrapper, SDXLTEUnetWrapper
from hcpdiff.models.compose import ComposeEmbPTHook, ComposeTEEXHook
from hcpdiff.models.compose import SDXLTextEncoder
from hcpdiff.utils.cfg_net_tools import make_hcpdiff, make_plugin
from hcpdiff.utils.net_utils import get_scheduler, auto_tokenizer_cls, auto_text_encoder_cls, load_emb
from hcpdiff.utils.utils import load_config_with_cli, get_cfg_range, mgcd, format_number
from hcpdiff.visualizer import Visualizer

def checkpoint_fix(function, *args, use_reentrant: bool = False, checkpoint_raw=torch.utils.checkpoint.checkpoint, **kwargs):
    return checkpoint_raw(function, *args, use_reentrant=use_reentrant, **kwargs)

torch.utils.checkpoint.checkpoint = checkpoint_fix

class Trainer:
    weight_dtype_map = {'fp32':torch.float32, 'fp16':torch.float16, 'bf16':torch.bfloat16}
    ckpt_manager_map = {'torch':CkptManagerPKL, 'safetensors':CkptManagerSafe}

    def __init__(self, cfgs_raw):
        cfgs_raw = TrainCFGConverter().convert(cfgs_raw)  # support old cfgs format
        cfgs = hydra.utils.instantiate(cfgs_raw)
        self.cfgs = cfgs

        self.init_context(cfgs_raw)
        self.build_loggers(cfgs_raw)

        self.train_TE = any([cfgs.text_encoder, cfgs.lora_text_encoder, cfgs.plugin_TE])

        self.build_ckpt_manager()
        self.build_model()
        self.make_hooks()
        self.config_model()
        self.cache_latents = False

        self.batch_size_list = []
        assert len(cfgs.data)>0, "At least one dataset is need."
        loss_weights = [dataset.keywords['loss_weight'] for name, dataset in cfgs.data.items()]
        self.train_loader_group = DataGroup([self.build_data(dataset) for name, dataset in cfgs.data.items()], loss_weights)

        if self.cache_latents:
            self.vae = self.vae.to('cpu')
        self.build_optimizer_scheduler()
        try:
            self.criterion = cfgs.train.loss.criterion(noise_scheduler=self.noise_scheduler, device=self.device)
        except:
            self.criterion = cfgs.train.loss.criterion()

        self.cfg_scale = get_cfg_range(cfgs.train.cfg_scale)
        if self.cfg_scale[1] == 1.0:
            self.cfg_context = CFGContext()
        else:  # DreamArtist
            self.cfg_context = DreamArtistPTContext(self.cfg_scale, self.num_train_timesteps)

        with torch.no_grad():
            self.build_ema()

        self.load_resume()

        torch.backends.cuda.matmul.allow_tf32 = cfgs.allow_tf32

        # calculate steps and epochs
        self.steps_per_epoch = len(self.train_loader_group.loader_list[0])
        if self.cfgs.train.train_epochs is not None:
            self.cfgs.train.train_steps = self.cfgs.train.train_epochs*self.steps_per_epoch
        else:
            self.cfgs.train.train_epochs = math.ceil(self.cfgs.train.train_steps/self.steps_per_epoch)

        if self.is_local_main_process and self.cfgs.previewer is not None:
            self.previewer = self.cfgs.previewer(exp_dir=self.exp_dir, te_hook=self.text_enc_hook, unet=self.TE_unet.unet,
                                                 TE=self.TE_unet.TE, tokenizer=self.tokenizer, vae=self.vae)

        self.prepare()

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_local_main_process(self):
        return self.accelerator.is_local_main_process

    def init_context(self, cfgs_raw):
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfgs.train.gradient_accumulation_steps,
            mixed_precision=self.cfgs.mixed_precision,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],  # fix inplace bug in DDP while use data_class
        )

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = self.accelerator.num_processes

        set_seed(self.cfgs.seed+self.local_rank)

    def build_loggers(self, cfgs_raw):
        if self.is_local_main_process:
            self.exp_dir = self.cfgs.exp_dir.format(time=time.strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(os.path.join(self.exp_dir, 'ckpts/'), exist_ok=True)
            with open(os.path.join(self.exp_dir, 'cfg.yaml'), 'w', encoding='utf-8') as f:
                f.write(OmegaConf.to_yaml(cfgs_raw))
            self.loggers: LoggerGroup = LoggerGroup([builder(exp_dir=self.exp_dir) for builder in self.cfgs.logger])
        else:
            self.loggers: LoggerGroup = LoggerGroup([builder(exp_dir=None) for builder in self.cfgs.logger])

        self.min_log_step = mgcd(*([item.log_step for item in self.loggers.logger_list]))
        image_log_steps = [item.image_log_step for item in self.loggers.logger_list if item.enable_log_image]
        if len(image_log_steps)>0:
            self.min_img_log_step = mgcd(*image_log_steps)
        else:
            self.min_img_log_step = -1

        self.loggers.info(f'world_size: {self.world_size}')
        self.loggers.info(f'accumulation: {self.cfgs.train.gradient_accumulation_steps}')

        if self.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_warning()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    def prepare(self):
        # Prepare everything with accelerator.
        prepare_name_list, prepare_obj_list = [], []
        if self.TE_unet.train_TE:
            prepare_obj_list.append(self.TE_unet)
            prepare_name_list.append('TE_unet')
        else:
            prepare_obj_list.append(self.TE_unet.unet)
            prepare_name_list.append('TE_unet.unet')

        if hasattr(self, 'optimizer'):
            prepare_obj_list.extend([self.optimizer, self.lr_scheduler] if self.lr_scheduler else [self.optimizer])
            prepare_name_list.extend(['optimizer', 'lr_scheduler'] if self.lr_scheduler else ['optimizer'])
        if hasattr(self, 'optimizer_pt'):
            prepare_obj_list.extend([self.optimizer_pt, self.lr_scheduler_pt] if self.lr_scheduler_pt else [self.optimizer_pt])
            prepare_name_list.extend(['optimizer_pt', 'lr_scheduler_pt'] if self.lr_scheduler_pt else ['optimizer_pt'])

        prepare_obj_list.extend(self.train_loader_group.loader_list)
        prepared_obj = self.accelerator.prepare(*prepare_obj_list)

        if not self.TE_unet.train_TE:
            self.TE_unet.unet = prepared_obj[0]
            prepared_obj = prepared_obj[1:]
            prepare_name_list = prepare_name_list[1:]

        ds_num = len(self.train_loader_group.loader_list)
        self.train_loader_group.loader_list = list(prepared_obj[-ds_num:])
        prepared_obj = prepared_obj[:-ds_num]

        for name, obj in zip(prepare_name_list, prepared_obj):
            setattr(self, name, obj)

        if self.cfgs.model.force_cast_precision:
            self.TE_unet.to(dtype=self.weight_dtype)

    def scale_lr(self, parameters):
        bs = sum(self.batch_size_list)
        scale_factor = bs*self.world_size*self.cfgs.train.gradient_accumulation_steps
        for param in parameters:
            if 'lr' in param:
                param['lr'] *= scale_factor

    def build_model(self):
        # Load the tokenizer
        if self.cfgs.model.get('tokenizer', None) is not None:
            self.tokenizer = self.cfgs.model.tokenizer
        else:
            tokenizer_cls = auto_tokenizer_cls(self.cfgs.model.pretrained_model_name_or_path, self.cfgs.model.revision)
            self.tokenizer = tokenizer_cls.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="tokenizer",
                revision=self.cfgs.model.revision, use_fast=False,
            )

        # Load scheduler and models
        self.noise_scheduler = self.cfgs.model.get('noise_scheduler', None) or \
                               DDPMScheduler.from_pretrained(self.cfgs.model.pretrained_model_name_or_path, subfolder='scheduler')

        self.num_train_timesteps = len(self.noise_scheduler.timesteps)
        self.vae: AutoencoderKL = self.cfgs.model.get('vae', None) or AutoencoderKL.from_pretrained(
            self.cfgs.model.pretrained_model_name_or_path, subfolder="vae", revision=self.cfgs.model.revision)
        self.build_unet_and_TE()

    def build_unet_and_TE(self):  # for easy to use colossalAI
        unet = self.cfgs.model.get('unet', None) or UNet2DConditionModel.from_pretrained(
            self.cfgs.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfgs.model.revision
        )

        if self.cfgs.model.get('text_encoder', None) is not None:
            text_encoder = self.cfgs.model.text_encoder
            text_encoder_cls = type(text_encoder)
        else:
            # import correct text encoder class
            text_encoder_cls = auto_text_encoder_cls(self.cfgs.model.pretrained_model_name_or_path, self.cfgs.model.revision)
            text_encoder = text_encoder_cls.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfgs.model.revision
            )

        # Wrap unet and text_encoder to make DDP happy. Multiple DDP has soooooo many fxxking bugs!
        wrapper_cls = SDXLTEUnetWrapper if text_encoder_cls == SDXLTextEncoder else TEUnetWrapper
        self.TE_unet = wrapper_cls(unet, text_encoder, train_TE=self.train_TE)

    def build_ema(self):
        if self.cfgs.model.ema is not None:
            self.ema_unet = self.cfgs.model.ema(self.TE_unet.unet)
            if self.train_TE:
                self.ema_text_encoder = self.cfgs.model.ema(self.TE_unet.TE)

    def build_ckpt_manager(self):
        self.ckpt_manager = self.ckpt_manager_map[self.cfgs.ckpt_type]()
        if self.is_local_main_process:
            self.ckpt_manager.set_save_dir(os.path.join(self.exp_dir, 'ckpts'), emb_dir=self.cfgs.tokenizer_pt.emb_dir)

    @property
    def unet_raw(self):
        return self.TE_unet.module.unet if self.train_TE else self.TE_unet.unet.module

    @property
    def TE_raw(self):
        return self.TE_unet.module.TE if self.train_TE else self.TE_unet.TE

    def config_model(self):
        if self.cfgs.model.enable_xformers:
            if is_xformers_available():
                self.TE_unet.unet.enable_xformers_memory_efficient_attention()
                # self.text_enc_hook.enable_xformers()
            else:
                warnings.warn("xformers is not available. Make sure it is installed correctly")

        self.vae.requires_grad_(False)
        self.TE_unet.requires_grad_(False)

        self.TE_unet.eval()

        if self.cfgs.model.gradient_checkpointing:
            self.TE_unet.enable_gradient_checkpointing()

        self.weight_dtype = self.weight_dtype_map.get(self.cfgs.mixed_precision, torch.float32)
        self.vae_dtype = self.weight_dtype_map.get(self.cfgs.model.get('vae_dtype', None), torch.float32)
        # Move vae and text_encoder to device and cast to weight_dtype
        self.vae = self.vae.to(self.device, dtype=self.vae_dtype)
        if not self.train_TE:
            self.TE_unet.TE = self.TE_unet.TE.to(self.device, dtype=self.weight_dtype)

    @torch.no_grad()
    def load_resume(self):
        if self.cfgs.train.resume is not None:
            for ckpt in self.cfgs.train.resume.ckpt_path.unet:
                self.ckpt_manager.load_ckpt_to_model(self.TE_unet.unet, ckpt, model_ema=getattr(self, 'ema_unet', None))
            for ckpt in self.cfgs.train.resume.ckpt_path.TE:
                self.ckpt_manager.load_ckpt_to_model(self.TE_unet.TE, ckpt, model_ema=getattr(self, 'ema_text_encoder', None))
            for name, ckpt in self.cfgs.train.resume.ckpt_path.words:
                self.ex_words_emb[name].data = load_emb(ckpt)

    def make_hooks(self):
        # Hook tokenizer and embedding to support pt
        self.embedding_hook, self.ex_words_emb = ComposeEmbPTHook.hook_from_dir(
            self.cfgs.tokenizer_pt.emb_dir, self.tokenizer, self.TE_unet.TE, log=self.is_local_main_process,
            N_repeats=self.cfgs.model.tokenizer_repeats, device=self.device)

        self.text_enc_hook = ComposeTEEXHook.hook(self.TE_unet.TE, self.tokenizer, N_repeats=self.cfgs.model.tokenizer_repeats,
                                                  device=self.device, clip_skip=self.cfgs.model.clip_skip,
                                                  clip_final_norm=self.cfgs.model.clip_final_norm)

    def build_dataset(self, data_builder: partial):
        batch_size = data_builder.keywords.pop('batch_size')
        cache_latents = data_builder.keywords.pop('cache_latents')
        self.batch_size_list.append(batch_size)

        train_dataset = data_builder(tokenizer=self.tokenizer, tokenizer_repeats=self.cfgs.model.tokenizer_repeats)
        train_dataset.bucket.build(batch_size*self.world_size, file_names=train_dataset.source.get_image_list())
        arb = isinstance(train_dataset.bucket, RatioBucket)
        self.loggers.info(f"len(train_dataset): {len(train_dataset)}")

        if cache_latents:
            self.cache_latents = True
            train_dataset.cache_latents(self.vae, self.vae_dtype, self.device, show_prog=self.is_local_main_process)
        return train_dataset, batch_size, arb

    def build_data(self, data_builder: partial) -> torch.utils.data.DataLoader:
        train_dataset, batch_size, arb = self.build_dataset(data_builder)

        # Pytorch Data loader
        train_sampler = get_sampler()(train_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=not arb)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=self.cfgs.train.workers,
                                                   sampler=train_sampler, collate_fn=train_dataset.collate_fn)
        return train_loader

    def get_param_group_train(self):
        # make miniFT and warp with lora
        self.DA_lora = False
        train_params_unet, self.lora_unet = make_hcpdiff(self.TE_unet.unet, self.cfgs.unet, self.cfgs.lora_unet)
        if isinstance(self.lora_unet, tuple):  # creat negative lora
            self.DA_lora = True
            self.lora_unet, self.lora_unet_neg = self.lora_unet
        train_params_unet_plugin, self.all_plugin_unet = make_plugin(self.TE_unet.unet, self.cfgs.plugin_unet)
        train_params_unet += train_params_unet_plugin

        if self.train_TE:
            train_params_text_encoder, self.lora_TE = make_hcpdiff(self.TE_unet.TE, self.cfgs.text_encoder, self.cfgs.lora_text_encoder)
            if isinstance(self.lora_TE, tuple):  # creat negative lora
                self.DA_lora = True
                self.lora_TE, self.lora_TE_neg = self.lora_TE
            train_params_TE_plugin, self.all_plugin_TE = make_plugin(self.TE_unet.TE, self.cfgs.plugin_TE)
            train_params_text_encoder += train_params_TE_plugin
        else:
            train_params_text_encoder = []

        N_params_unet = format_number(sum(sum(x.numel() for x in p['params']) for p in train_params_unet))
        N_params_TE = format_number(sum(sum(x.numel() for x in p['params']) for p in train_params_text_encoder))
        self.loggers.info(f'unet trainable params: {N_params_unet}; text encoder trainable params: {N_params_TE}')

        # params for embedding
        train_params_emb = []
        self.train_pts = {}
        if self.cfgs.tokenizer_pt.train is not None:
            for v in self.cfgs.tokenizer_pt.train:
                word_emb = self.ex_words_emb[v.name]
                self.train_pts[v.name] = word_emb
                word_emb.requires_grad = True
                self.embedding_hook.emb_train.append(word_emb)
                train_params_emb.append({'params':word_emb, 'lr':v.lr})

        return train_params_unet+train_params_text_encoder, train_params_emb

    def build_optimizer_scheduler(self):
        # set optimizer
        parameters, parameters_pt = self.get_param_group_train()

        if len(parameters)>0:  # do fine-tuning
            cfg_opt = self.cfgs.train.optimizer
            if self.cfgs.train.scale_lr:
                self.scale_lr(parameters)
            assert isinstance(cfg_opt, partial), f'optimizer.type is not supported anymore, please use class path like "torch.optim.AdamW".'
            self.optimizer = cfg_opt(params=parameters)
            self.lr_scheduler = get_scheduler(self.cfgs.train.scheduler, self.optimizer)

        if len(parameters_pt)>0:  # do prompt-tuning
            cfg_opt_pt = self.cfgs.train.optimizer_pt
            if self.cfgs.train.scale_lr_pt:
                self.scale_lr(parameters_pt)
            assert isinstance(cfg_opt_pt, partial), f'optimizer.type is not supported anymore, please use class path like "torch.optim.AdamW".'
            self.optimizer_pt = cfg_opt_pt(params=parameters_pt)
            self.lr_scheduler_pt = get_scheduler(self.cfgs.train.scheduler_pt, self.optimizer_pt)

    def train(self, loss_ema=0.93):
        total_batch_size = sum(self.batch_size_list)*self.world_size*self.cfgs.train.gradient_accumulation_steps

        self.loggers.info("***** Running training *****")
        self.loggers.info(f"  Num batches each epoch = {len(self.train_loader_group.loader_list[0])}")
        self.loggers.info(f"  Num Steps = {self.cfgs.train.train_steps}")
        self.loggers.info(f"  Instantaneous batch size per device = {sum(self.batch_size_list)}")
        self.loggers.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.loggers.info(f"  Gradient Accumulation steps = {self.cfgs.train.gradient_accumulation_steps}")
        self.global_step = 0
        if self.cfgs.train.resume is not None:
            self.global_step = self.cfgs.train.resume.start_step

        loss_sum = None
        for data_list in self.train_loader_group:
            loss = self.train_one_step(data_list)
            loss_sum = loss if loss_sum is None else (loss_ema*loss_sum+(1-loss_ema)*loss)

            self.global_step += 1
            if self.is_local_main_process:
                if self.global_step%self.cfgs.train.save_step == 0:
                    self.save_model()
                if self.global_step%self.min_log_step == 0:
                    # get learning rate from optimizer
                    lr_model = self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else 0.
                    lr_word = self.optimizer_pt.param_groups[0]['lr'] if hasattr(self, 'optimizer_pt') else 0.
                    self.loggers.log(datas={
                        'Step':{'format':'[{}/{}]', 'data':[self.global_step, self.cfgs.train.train_steps]},
                        'Epoch':{'format':'[{}/{}]<{}/{}>', 'data':[self.global_step//self.steps_per_epoch, self.cfgs.train.train_epochs,
                            self.global_step%self.steps_per_epoch, self.steps_per_epoch]},
                        'LR_model':{'format':'{:.2e}', 'data':[lr_model]},
                        'LR_word':{'format':'{:.2e}', 'data':[lr_word]},
                        'Loss':{'format':'{:.5f}', 'data':[loss_sum]},
                    }, step=self.global_step)
                if self.min_img_log_step>0 and self.global_step%self.min_img_log_step == 0:
                    self.loggers.log_image(self.previewer.preview_dict(), self.global_step)

            if self.global_step>=self.cfgs.train.train_steps:
                break

        self.wait_for_everyone()
        if self.is_local_main_process:
            self.save_model()

    def wait_for_everyone(self):
        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def get_latents(self, image, dataset):
        if dataset.latents is None:
            latents = self.vae.encode(image.to(dtype=self.vae.dtype)).latent_dist.sample()
            latents = latents*self.vae.config.scaling_factor
        else:
            latents = image  # Cached latents
        return latents

    def make_noise(self, latents):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        return self.noise_scheduler.add_noise(latents, noise, timesteps), noise, timesteps

    def forward(self, latents, prompt_ids, attn_mask=None, position_ids=None, **kwargs):
        noisy_latents, noise, timesteps = self.make_noise(latents)

        # CFG context for DreamArtist
        noisy_latents, timesteps = self.cfg_context.pre(noisy_latents, timesteps)
        model_pred = self.TE_unet(prompt_ids, noisy_latents, timesteps, attn_mask=attn_mask, position_ids=position_ids, **kwargs)
        model_pred = self.cfg_context.post(model_pred)

        # Get the target for loss depending on the prediction type
        if self.cfgs.train.loss.type == "eps":
            target = noise
        elif self.cfgs.train.loss.type == "sample":
            target = self.noise_scheduler.step(noise, timesteps, noisy_latents)
            model_pred = self.noise_scheduler.step(model_pred, timesteps, noisy_latents)
        else:
            raise ValueError(f"Unknown loss type {self.cfgs.train.loss.type}")
        return model_pred, target, timesteps

    def train_one_step(self, data_list):
        with self.accelerator.accumulate(self.TE_unet):
            for idx, data in enumerate(data_list):
                image = data.pop('img').to(self.device, dtype=self.weight_dtype)
                img_mask = data.pop('mask').to(self.device) if 'mask' in data else None
                prompt_ids = data.pop('prompt').to(self.device)
                attn_mask = data.pop('attn_mask').to(self.device) if 'attn_mask' in data else None
                position_ids = data.pop('position_ids').to(self.device) if 'position_ids' in data else None
                other_datas = {k:v.to(self.device) for k, v in data.items() if k != 'plugin_input'}
                if 'plugin_input' in data:
                    other_datas['plugin_input'] = {k:v.to(self.device, dtype=self.weight_dtype) for k, v in data['plugin_input'].items()}

                latents = self.get_latents(image, self.train_loader_group.get_dataset(idx))
                model_pred, target, timesteps = self.forward(latents, prompt_ids, attn_mask, position_ids, **other_datas)
                loss = self.get_loss(model_pred, target, timesteps, img_mask)*self.train_loader_group.get_loss_weights(idx)
                self.accelerator.backward(loss)

            if hasattr(self, 'optimizer'):
                if self.accelerator.sync_gradients:  # fine-tuning
                    if hasattr(self.TE_unet, 'trainable_parameters'):
                        clip_param = self.TE_unet.trainable_parameters()
                    else:
                        clip_param = self.TE_unet.module.trainable_parameters()
                    self.accelerator.clip_grad_norm_(clip_param, self.cfgs.train.max_grad_norm)
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=self.cfgs.train.set_grads_to_none)

            if hasattr(self, 'optimizer_pt'):  # prompt tuning
                self.optimizer_pt.step()
                if self.lr_scheduler_pt:
                    self.lr_scheduler_pt.step()
                self.optimizer_pt.zero_grad(set_to_none=self.cfgs.train.set_grads_to_none)

            if self.accelerator.sync_gradients:
                self.update_ema()
        return loss.item()

    def get_loss(self, model_pred, target, timesteps, att_mask):
        if att_mask is None:
            att_mask = 1.0
        if getattr(self.criterion, 'need_timesteps', False):
            loss = (self.criterion(model_pred.float(), target.float(), timesteps)*att_mask).mean()
        else:
            loss = (self.criterion(model_pred.float(), target.float())*att_mask).mean()
        if len(self.embedding_hook.emb_train)>0:
            loss = loss+0*sum([emb.mean() for emb in self.embedding_hook.emb_train])
        return loss

    def update_ema(self):
        if hasattr(self, 'ema_unet'):
            self.ema_unet.update(self.unet_raw)
        if hasattr(self, 'ema_text_encoder'):
            self.ema_text_encoder.update(self.TE_raw)

    def save_model(self, from_raw=False):
        unet_raw = self.unet_raw
        self.ckpt_manager.save_model_with_lora(unet_raw, self.lora_unet, model_ema=getattr(self, 'ema_unet', None),
                                               name='unet', step=self.global_step)
        self.ckpt_manager.save_plugins(unet_raw, self.all_plugin_unet, name='unet', step=self.global_step,
                                       model_ema=getattr(self, 'ema_unet', None))
        if self.train_TE:
            TE_raw = self.TE_raw
            # exclude_key: embeddings should not save with text-encoder
            self.ckpt_manager.save_model_with_lora(TE_raw, self.lora_TE, model_ema=getattr(self, 'ema_text_encoder', None),
                                                   name='text_encoder', step=self.global_step, exclude_key='emb_ex.')
            self.ckpt_manager.save_plugins(TE_raw, self.all_plugin_TE, name='text_encoder', step=self.global_step,
                                           model_ema=getattr(self, 'ema_text_encoder', None))

        if self.DA_lora:
            self.ckpt_manager.save_model_with_lora(None, self.lora_unet_neg, name='unet-neg', step=self.global_step)
            if self.train_TE:
                self.ckpt_manager.save_model_with_lora(None, self.lora_TE_neg, name='text_encoder-neg', step=self.global_step)

        self.ckpt_manager.save_embedding(self.train_pts, self.global_step, self.cfgs.tokenizer_pt.replace)

        self.loggers.info(f"Saved state, step: {self.global_step}")

    def make_vis(self):
        vis_dir = os.path.join(self.exp_dir, f'vis-{self.global_step}')
        new_components = {
            'unet':self.unet_raw,
            'text_encoder':self.TE_raw,
            'tokenizer':self.tokenizer,
            'vae':self.vae,
        }
        viser = Visualizer(self.cfgs.model.pretrained_model_name_or_path, new_components=new_components)
        if self.cfgs.vis_info.prompt:
            raise ValueError('vis_info.prompt is None. cannot generate without prompt.')
        viser.vis_to_dir(vis_dir, self.cfgs.vis_prompt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = Trainer(conf)
    trainer.train()
