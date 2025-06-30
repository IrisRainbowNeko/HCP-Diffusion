from pathlib import Path

import torch
from rainbowneko.evaluate.preview import WorkflowPreviewer
from rainbowneko.utils import to_cuda

from hcpdiff.models.wrapper import SD15Wrapper
from accelerate.hooks import remove_hook_from_module
from typing import Dict
from types import ModuleType

class HCPPreviewer(WorkflowPreviewer):
    def __init__(self, parser, cfgs_raw, workflow: str | ModuleType | Dict, ds_name=None, interval=100, trainer=None,
                 mixed_precision=None, seed=42, **cfgs):
        super().__init__(parser, cfgs_raw, workflow, ds_name=ds_name, interval=interval, trainer=trainer,
                         mixed_precision=mixed_precision, seed=seed, **cfgs)
        if trainer is None:
            self.pt_trainable = False
        else:
            self.emb_pt = trainer.cfgs.emb_pt
            self.pt_trainable = trainer.pt_trainable

    @torch.no_grad()
    def evaluate(self, step: int, prefix='eval/'):
        if step%self.interval != 0 or not self.is_local_main_process:
            return

        # record training layers
        if self.model_wrapper is not None:
            training_layers = [layer for layer in self.model_raw.modules() if layer.training]
            self.model_wrapper.eval()
            model = self.model_raw
        else:
            training_layers = []
            model = None

        if self.loggers is not None:
            self.loggers.info(f'Preview')

        N_repeats = model.text_enc_hook.N_repeats
        clip_skip = model.text_enc_hook.clip_skip
        clip_final_norm = model.text_enc_hook.clip_final_norm
        use_attention_mask = model.text_enc_hook.use_attention_mask

        preview_root = Path(self.exp_dir)/'imgs'
        preview_root.mkdir(parents=True, exist_ok=True)

        states = self.workflow_runner.run(model=model, in_preview=True, te_hook=model.text_enc_hook,
                                          device=self.device, dtype=self.weight_dtype, preview_root=preview_root, preview_step=step,
                                          world_size=self.world_size, local_rank=self.local_rank,
                                          emb_hook=self.emb_pt.embedding_hook if self.pt_trainable else None)

        # restore model states
        if model.vae is not None:
            model.vae.disable_tiling()
            model.vae.disable_slicing()
            remove_hook_from_module(model.vae, recurse=True)
            if 'vae_encode_raw' in states:
                model.vae.encode = states['vae_encode_raw']
                model.vae.decode = states['vae_decode_raw']

        if 'emb_hook' in states and not self.pt_trainable:
            states['emb_hook'].remove()

        if self.pt_trainable:
            self.emb_pt.embedding_hook.N_repeats = N_repeats

        model.tokenizer.N_repeats = N_repeats
        model.text_enc_hook.N_repeats = N_repeats
        model.text_enc_hook.clip_skip = clip_skip
        model.text_enc_hook.clip_final_norm = clip_final_norm
        model.text_enc_hook.use_attention_mask = use_attention_mask
        
        to_cuda(model)

        for layer in training_layers:
            layer.train()
