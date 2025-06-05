from pathlib import Path

import torch
from accelerate.hooks import remove_hook_from_module
from rainbowneko.evaluate import WorkflowEvaluator, MetricGroup
from rainbowneko.utils import to_cuda

from hcpdiff.models.wrapper import SD15Wrapper

class HCPEvaluator(WorkflowEvaluator):

    @torch.no_grad()
    def evaluate(self, step: int, model: SD15Wrapper, prefix='eval/'):
        if step%self.interval != 0 or not self.trainer.is_local_main_process:
            return

        # record training layers
        training_layers = [layer for layer in model.modules() if layer.training]

        model.eval()
        self.trainer.loggers.info(f'Preview')

        N_repeats = model.text_enc_hook.N_repeats
        clip_skip = model.text_enc_hook.clip_skip
        clip_final_norm = model.text_enc_hook.clip_final_norm
        use_attention_mask = model.text_enc_hook.use_attention_mask

        preview_root = Path(self.trainer.exp_dir)/'imgs'
        preview_root.mkdir(parents=True, exist_ok=True)

        states = self.workflow_runner.run(model=model, in_preview=True, te_hook=model.text_enc_hook,
                                          device=self.device, dtype=self.dtype, preview_root=preview_root, preview_step=step,
                                          world_size=self.trainer.world_size, local_rank=self.trainer.local_rank,
                                          emb_hook=self.trainer.cfgs.emb_pt.embedding_hook if self.trainer.pt_trainable else None)

        # get metrics
        metric = states['_metric']

        v_metric = metric.finish(self.trainer.accelerator.gather, self.trainer.is_local_main_process)
        if not isinstance(v_metric, dict):
            v_metric = {'metric':v_metric}

        log_data = {
            "eval/Step":{
                "format":"{}",
                "data":[step],
            }
        }
        log_data.update(MetricGroup.format(v_metric, prefix=prefix))
        self.trainer.loggers.log(log_data, step, force=True)

        # restore model states
        if model.vae is not None:
            model.vae.disable_tiling()
            model.vae.disable_slicing()
            remove_hook_from_module(model.vae, recurse=True)
            if 'vae_encode_raw' in states:
                model.vae.encode = states['vae_encode_raw']
                model.vae.decode = states['vae_decode_raw']

        if 'emb_hook' in states and not self.trainer.pt_trainable:
            states['emb_hook'].remove()

        if self.trainer.pt_trainable:
            self.trainer.cfgs.emb_pt.embedding_hook.N_repeats = N_repeats

        model.tokenizer.N_repeats = N_repeats
        model.text_enc_hook.N_repeats = N_repeats
        model.text_enc_hook.clip_skip = clip_skip
        model.text_enc_hook.clip_final_norm = clip_final_norm
        model.text_enc_hook.use_attention_mask = use_attention_mask

        to_cuda(model)

        for layer in training_layers:
            layer.train()
