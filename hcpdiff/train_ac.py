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
import itertools
import os
import sys

import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import hydra
from loguru import logger
import time

import diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available

from hcpdiff.data import TextImagePairDataset, RatioBucket
from hcpdiff.utils.utils import get_scheduler, import_model_class_from_model_name_or_path, cycle_data,\
    load_config_with_cli, get_cfg_range
from hcpdiff.models import EmbeddingPTHook, TEEXHook, CFGContext, DreamArtistPTContext
from hcpdiff.utils.ema import ModelEMA
from hcpdiff.utils.cfg_net_tools import make_hcpdiff
from hcpdiff.utils.emb_utils import load_emb
from hcpdiff.data import collate_fn_ft
from hcpdiff.visualizer import Visualizer
from hcpdiff.utils.ckpt_manager import CkptManagerPKL, CkptManagerSafe

class Trainer:
    def __init__(self, cfgs_raw):
        cfgs = hydra.utils.instantiate(cfgs_raw)
        self.cfgs=cfgs

        self.init_context(cfgs_raw)

        if self.is_local_main_process:
            self.exp_dir = os.path.join(self.cfgs.exp_dir, f'{time.strftime("%Y-%m-%d-%H-%M-%S")}')
            os.makedirs(os.path.join(self.exp_dir, 'ckpts/'), exist_ok=True)
            logger.add(os.path.join(self.exp_dir, 'train.log'))
            with open(os.path.join(self.exp_dir, 'cfg.yaml'), 'w', encoding='utf-8') as f:
                f.write(OmegaConf.to_yaml(cfgs_raw))
        else:
            logger.disable("__main__")

        logger.info(f'world_size: {self.world_size}')
        logger.info(f'accumulation: {self.cfgs.train.gradient_accumulation_steps}')

        if self.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_warning()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        self.lr=1e-5 # no usage, place set lr in cfgs
        self.train_TE = (cfgs.text_encoder is not None) or (cfgs.lora_text_encoder is not None)

        self.build_ckpt_manager()
        self.build_model()
        self.make_hooks()
        self.config_model()
        self.train_loader, self.arb_ist=self.build_data(cfgs.data)
        if cfgs.data_class is None:
            self.train_loader_class=None # without DreamBooth
        else:
            self.train_loader_class, self.arb_class=self.build_data(cfgs.data_class)
        if self.cache_latents:
            self.vae = self.vae.to('cpu')
        self.build_optimizer_scheduler()
        self.criterion = cfgs.train.loss.criterion

        self.cfg_scale = get_cfg_range(cfgs.train.cfg_scale)
        if self.cfg_scale[1]==1.0:
            self.cfg_context = CFGContext()
        else: # DreamArtist
            self.cfg_context = DreamArtistPTContext(self.cfg_scale, self.num_train_timesteps)

        with torch.no_grad():
            self.build_ema()

        self.load_resume()

        if cfgs.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.prepare()

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_local_main_process(self):
        return self.accelerator.is_local_main_process

    def init_context(self, cfgs_raw):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfgs.train.gradient_accumulation_steps,
            mixed_precision=self.cfgs.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = self.accelerator.num_processes

        set_seed(self.cfgs.seed + self.local_rank)

    def prepare(self):
        # Prepare everything with accelerator.
        prepare_obj_list = [self.unet, self.train_loader]
        prepare_name_list = ['unet', 'train_loader']
        if hasattr(self, 'optimizer'):
            prepare_obj_list.extend([self.optimizer, self.lr_scheduler])
            prepare_name_list.extend(['optimizer', 'lr_scheduler'])
        if self.train_TE:
            prepare_obj_list.append(self.text_encoder)
            prepare_name_list.append('text_encoder')
        if hasattr(self, 'optimizer_pt'):
            prepare_obj_list.extend([self.optimizer_pt, self.lr_scheduler_pt])
            prepare_name_list.extend(['optimizer_pt', 'lr_scheduler_pt'])

        prepared_obj = self.accelerator.prepare(*prepare_obj_list)
        for name, obj in zip(prepare_name_list, prepared_obj):
            setattr(self, name, obj)

    def scale_lr(self, parameters):
        bs = self.cfgs.data.batch_size
        if self.cfgs.data_class is not None:
            bs += self.cfgs.data_class.batch_size
        scale_factor = bs * self.world_size * self.cfgs.train.gradient_accumulation_steps
        for param in parameters:
            if 'lr' in param:
                param['lr'] *= scale_factor

    def build_model(self):
        # Load the tokenizer
        if self.cfgs.model.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfgs.model.tokenizer_name, revision=self.cfgs.model.revision, use_fast=False)
        elif self.cfgs.model.pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="tokenizer",
                revision=self.cfgs.model.revision, use_fast=False,
            )

        # Load scheduler and models
        self.noise_scheduler = getattr(diffusers, self.cfgs.model.noise_scheduler).from_pretrained(self.cfgs.model.pretrained_model_name_or_path, subfolder="scheduler")
        self.num_train_timesteps = len(self.noise_scheduler.timesteps)
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(self.cfgs.model.pretrained_model_name_or_path, subfolder="vae",
                                            revision=self.cfgs.model.revision)
        self.build_unet_and_TE()

    def build_unet_and_TE(self): # for easy to use colossalAI
        self.unet = UNet2DConditionModel.from_pretrained(
            self.cfgs.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfgs.model.revision
        )
        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(self.cfgs.model.pretrained_model_name_or_path,
                                                                      self.cfgs.model.revision)
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.cfgs.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfgs.model.revision
        )


    def build_ema(self):
        if self.cfgs.model.ema_unet>0:
            self.ema_unet = ModelEMA(self.unet.named_parameters(), self.cfgs.model.ema_unet)
        if self.train_TE and self.cfgs.model.ema_text_encoder>0:
            self.ema_text_encoder = ModelEMA(self.text_encoder.named_parameters(), self.cfgs.model.ema_text_encoder)

    def build_ckpt_manager(self):
        if self.cfgs.ckpt_type=='torch':
            self.ckpt_manager = CkptManagerPKL()
        elif self.cfgs.ckpt_type=='safetensors':
            self.ckpt_manager = CkptManagerSafe()
        else:
            raise NotImplementedError(f'Not support ckpt type: {self.cfgs.ckpt_type}')
        if self.is_local_main_process:
            self.ckpt_manager.set_save_dir(os.path.join(self.exp_dir, 'ckpts'), emb_dir=self.cfgs.tokenizer_pt.emb_dir)

    def get_unet_raw(self):
        return self.unet.module

    def get_text_encoder_raw(self):
        return self.text_encoder.module if self.train_TE else self.text_encoder

    def config_model(self):
        if self.cfgs.model.enable_xformers:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
                #self.text_enc_hook.enable_xformers()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.unet.eval()
        self.text_encoder.eval()

        if self.cfgs.model.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.train_TE:
                self.text_encoder.gradient_checkpointing_enable()

        weight_dtype = torch.float32
        if self.cfgs.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.cfgs.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        # Move vae and text_encoder to device and cast to weight_dtype
        self.vae.to(self.device, dtype=weight_dtype)
        if not self.train_TE:
            self.text_encoder.to(self.device, dtype=weight_dtype)

    @torch.no_grad()
    def load_resume(self):
        if self.cfgs.train.resume is not None:
            for ckpt in self.cfgs.train.resume.ckpt_path.unet:
                self.ckpt_manager.load_ckpt_to_model(self.unet, ckpt, model_ema=getattr(self, 'ema_unet', None))
            for ckpt in self.cfgs.train.resume.ckpt_path.TE:
                self.ckpt_manager.load_ckpt_to_model(self.text_encoder, ckpt, model_ema=getattr(self, 'ema_text_encoder', None))
            for name, ckpt in self.cfgs.train.resume.ckpt_path.words:
                self.ex_words_emb[name].data = load_emb(ckpt)

    def make_hooks(self):
        # Hook tokenizer and embedding to support pt
        self.embedding_hook, self.ex_words_emb = EmbeddingPTHook.hook_from_dir(
                        self.cfgs.tokenizer_pt.emb_dir, self.tokenizer, self.text_encoder, log=self.is_local_main_process,
                        N_repeats=self.cfgs.model.tokenizer_repeats, device=self.device)

        self.text_enc_hook = TEEXHook(self.text_encoder, self.tokenizer, N_repeats=self.cfgs.model.tokenizer_repeats, device=self.device,
                                      clip_skip=self.cfgs.model.clip_skip)

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
            train_dataset.cache_latents(self.vae, self.weight_dtype, show_prog=self.is_local_main_process)

        # Pytorch Data loader
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=self.world_size,
                                                                        rank=self.local_rank, shuffle=not arb)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg_data.batch_size,
            num_workers=self.cfgs.train.workers, sampler=train_sampler, collate_fn=collate_fn_ft)
        return train_loader, arb

    def get_param_group_train(self):
        # make miniFT and warp with lora
        self.DA_lora = False
        train_params_unet, self.lora_unet = make_hcpdiff(self.unet, self.cfgs.unet, self.cfgs.lora_unet)
        if isinstance(self.lora_unet, tuple): # creat negative lora
            self.DA_lora = True
            self.lora_unet, self.lora_unet_neg = self.lora_unet

        if self.train_TE:
            train_params_text_encoder, self.lora_TE = make_hcpdiff(self.text_encoder, self.cfgs.text_encoder, self.cfgs.lora_text_encoder)
            if isinstance(self.lora_TE, tuple):  # creat negative lora
                self.DA_lora = True
                self.lora_TE, self.lora_TE_neg = self.lora_TE
        else:
            train_params_text_encoder=[]

        # params for embedding
        train_params_emb = []
        self.train_pts = {}
        if self.cfgs.tokenizer_pt.train is not None:
            for v in self.cfgs.tokenizer_pt.train:
                self.train_pts[v.name]=self.ex_words_emb[v.name]
                self.ex_words_emb[v.name].requires_grad=True
                train_params_emb.append({'params':self.ex_words_emb[v.name], 'lr':v.lr})

        return train_params_unet + train_params_text_encoder, train_params_emb

    def build_optimizer_scheduler(self):
        # set optimizer
        parameters, parameters_pt = self.get_param_group_train()

        cfg_opt = self.cfgs.train.optimizer
        if len(parameters)>0: # do fine-tuning
            if self.cfgs.train.scale_lr:
                self.scale_lr(parameters)

            if cfg_opt.type=='adamw_8bit':
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(params=parameters, lr=self.lr, weight_decay=cfg_opt.weight_decay)
            elif cfg_opt.type=='deepspeed' and self.accelerator.state.deepspeed_plugin is not None:
                from deepspeed.ops.adam import FusedAdam
                self.optimizer = FusedAdam(params=parameters, lr=self.lr, weight_decay=cfg_opt.weight_decay)
            elif cfg_opt.type=='adamw':
                self.optimizer = torch.optim.AdamW(params=parameters, lr=self.lr, weight_decay=cfg_opt.weight_decay)
            else:
                self.optimizer = cfg_opt.optimizer.opt(parameters, lr=self.lr)

            self.lr_scheduler = get_scheduler(optimizer=self.optimizer, **self.cfgs.train.scheduler)

        if len(parameters_pt)>0: # do prompt-tuning
            if self.cfgs.train.scale_lr_pt:
                self.scale_lr(parameters_pt)

            self.optimizer_pt = torch.optim.AdamW(params=parameters_pt, lr=self.lr, weight_decay=cfg_opt.weight_decay_pt)
            self.lr_scheduler_pt = get_scheduler(optimizer=self.optimizer_pt, **self.cfgs.train.scheduler_pt)

    def train(self):
        total_batch_size = self.cfgs.data.batch_size * self.world_size * self.cfgs.train.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num batches each epoch = {len(self.train_loader)}")
        logger.info(f"  Num Steps = {self.cfgs.train.scheduler.num_training_steps}")
        logger.info(f"  Instantaneous batch size per device = {self.cfgs.data.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.cfgs.train.gradient_accumulation_steps}")
        self.global_step = 0
        if self.cfgs.train.resume is not None:
            self.global_step = self.cfgs.train.resume.start_step

        if self.train_loader_class is not None:
            self.data_iter_class = iter(cycle_data(self.train_loader_class, arb=self.arb_class))

        loss_sum=0
        for image, att_mask, prompt_ids in cycle_data(self.train_loader, arb=self.arb_ist):
            loss=self.train_one_step(image, att_mask, prompt_ids)
            loss_sum+=loss

            self.global_step += 1
            if self.is_local_main_process:
                if self.global_step % self.cfgs.train.save_step == 0:
                    self.save_model()
                if self.global_step % self.cfgs.train.log_step == 0:
                    lr_model = self.lr_scheduler.get_last_lr()[0] if hasattr(self, 'lr_scheduler') else 0.
                    lr_word = self.lr_scheduler_pt.get_last_lr()[0] if hasattr(self, 'lr_scheduler_pt') else 0.
                    logger.info('Step [{}/{}], LR_model: {:.2e}, LR_word: {:.2e}, Loss: {:.5f}'
                                .format(self.global_step, self.cfgs.train.scheduler.num_training_steps,
                                        lr_model, lr_word, loss_sum / self.cfgs.train.log_step))
                    loss_sum = 0

            if self.global_step >= self.cfgs.train.scheduler.num_training_steps:
                break

        self.wait_for_everyone()
        self.save_model()

    def wait_for_everyone(self):
        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def get_latents(self, image, dataset):
        if dataset.latents is None:
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * 0.18215
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

    def encode_decode(self, prompt_ids, noisy_latents, timesteps):
        # for colossalAI support
        encoder_hidden_states = self.text_encoder(prompt_ids, output_hidden_states=True)  # Get the text embedding for conditioning
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample  # Predict the noise residual
        return model_pred

    def forward(self, latents, prompt_ids):
        noisy_latents, noise, timesteps = self.make_noise(latents)

        # CFG context for DreamArtist
        noisy_latents, timesteps = self.cfg_context.pre(noisy_latents, timesteps)
        model_pred = self.encode_decode(prompt_ids, noisy_latents, timesteps)
        model_pred = self.cfg_context.post(model_pred)

        # Get the target for loss depending on the prediction type
        if self.cfgs.train.loss.type == "eps":
            target = noise
        elif self.cfgs.train.loss.type == "sample":
            target = self.noise_scheduler.step(noise, timesteps, noisy_latents)
            model_pred = self.noise_scheduler.step(model_pred, timesteps, noisy_latents)
        else:
            raise ValueError(f"Unknown loss type {self.cfgs.train.loss.type}")
        return model_pred, target

    def train_one_step(self, image, att_mask, prompt_ids):
        image=image.to(self.device, dtype=self.weight_dtype)
        att_mask = att_mask.to(self.device)
        prompt_ids=prompt_ids.to(self.device)
        with self.accelerator.accumulate(self.unet):
            latents = self.get_latents(image, self.train_loader.dataset)
            model_pred, target = self.forward(latents, prompt_ids)

            if self.train_loader_class is not None:
                #DreamBooth prior forward
                image_cls, att_mask_cls, prompt_ids_cls = next(self.data_iter_class)
                image_cls = image_cls.to(self.device, dtype=self.weight_dtype)
                att_mask_cls = att_mask_cls.to(self.device)
                prompt_ids_cls = prompt_ids_cls.to(self.device)
                latents_cls = self.get_latents(image_cls, self.train_loader_class.dataset)
                model_pred_prior, target_prior = self.forward(latents_cls, prompt_ids_cls)

                loss = self.get_loss(model_pred, target, att_mask) # Compute instance loss
                prior_loss = self.get_loss(model_pred_prior, target_prior, att_mask_cls) # Compute prior loss
                loss = loss + self.cfgs.train.loss.prior_loss_weight * prior_loss
            else:
                loss = self.get_loss(model_pred, target, att_mask)

            self.accelerator.backward(loss)

            if hasattr(self, 'optimizer'):
                if self.accelerator.sync_gradients: # fine-tuning
                    params_to_clip = (
                        itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                        if self.train_TE else self.unet.parameters()
                    )
                    self.accelerator.clip_grad_norm_(params_to_clip, self.cfgs.train.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=self.cfgs.train.set_grads_to_none)

            if hasattr(self, 'optimizer_pt'): # prompt tuning
                self.optimizer_pt.step()
                self.lr_scheduler_pt.step()
                self.optimizer_pt.zero_grad(set_to_none=self.cfgs.train.set_grads_to_none)

            self.update_ema()
        return loss.item()

    def get_loss(self, model_pred, target, att_mask):
        return (self.criterion(model_pred.float(), target.float()) * att_mask).mean()

    def update_ema(self):
        if hasattr(self, 'ema_unet'):
            self.ema_unet.step(self.get_unet_raw().named_parameters())
        if hasattr(self, 'ema_text_encoder'):
            self.ema_text_encoder.step(self.get_text_encoder_raw().named_parameters())

    def save_model(self, from_raw=False):
        unet_raw=self.get_unet_raw()
        self.ckpt_manager.save_model_with_lora(unet_raw, self.lora_unet, model_ema=getattr(self, 'ema_unet', None),
                                               name='unet', step=self.global_step)
        if self.train_TE:
            TE_raw = self.get_text_encoder_raw()
            self.ckpt_manager.save_model_with_lora(TE_raw, self.lora_TE, model_ema=getattr(self, 'ema_text_encoder', None),
                                                   name='text_encoder', step=self.global_step)

        if self.DA_lora:
            self.ckpt_manager.save_model_with_lora(None, self.lora_unet_neg, name='unet-neg', step=self.global_step)
            if self.train_TE:
                self.ckpt_manager.save_model_with_lora(None, self.lora_TE_neg, name='text_encoder-neg', step=self.global_step)

        self.ckpt_manager.save_embedding(self.train_pts, self.global_step, self.cfgs.tokenizer_pt.replace)

        logger.info(f"Saved state, step: {self.global_step}")

    def make_vis(self):
        vis_dir = os.path.join(self.exp_dir ,f'vis-{self.global_step}')
        new_components={
            'unet': self.get_unet_raw(),
            'text_encoder': self.get_text_encoder_raw(),
            'tokenizer': self.tokenizer,
            'vae': self.vae,
        }
        viser = Visualizer(self.cfgs.model.pretrained_model_name_or_path, new_components=new_components)
        if self.cfgs.vis_info.prompt:
            raise ValueError('vis_info.prompt is None. cannot generate without prompt.')
        viser.vis_to_dir(vis_dir, self.cfgs.vis_prompt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, _ = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=sys.argv[3:]) # skip --cfg
    trainer=Trainer(conf)
    trainer.train()