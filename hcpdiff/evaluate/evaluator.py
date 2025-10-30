import os
from pathlib import Path
from types import ModuleType
from typing import Dict

import torch
from accelerate.hooks import remove_hook_from_module
from rainbowneko.evaluate import WorkflowEvaluator, MetricGroup
from rainbowneko.loggers import ScalarLog
from rainbowneko.utils import to_cuda

from hcpdiff.models.wrapper import SD15Wrapper

class HCPEvaluator(WorkflowEvaluator):
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
    def evaluate(self, step: int, model: SD15Wrapper, prefix='eval/'):
        if step%self.interval != 0 or not self.is_local_main_process:
            return

        # record training layers
        training_layers = [layer for layer in model.modules() if layer.training]

        model.eval()
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

        # get metrics
        metric = states['_metric']
        loggers = states.get('loggers', None)

        v_metric = metric.finish(self.accelerator.gather, self.is_local_main_process)
        if not isinstance(v_metric, dict):
            v_metric = {'metric':v_metric}

        log_data = {
            "eval/Step":ScalarLog(value=step, format="{}")
        }
        log_data.update(MetricGroup.format(v_metric, prefix=prefix))
        if self.loggers is not None:
            self.loggers.log(log_data, step, force=True)
        elif loggers is not None:
            loggers.log(log_data, step, force=True)
        else:
            print(', '.join([f"{os.path.basename(k)} = {v.format.format(*v.value)}" for k, v in log_data.items()]))

        if '_logs' in states:
            self.loggers.log(states['_logs'], step, force=True)

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
