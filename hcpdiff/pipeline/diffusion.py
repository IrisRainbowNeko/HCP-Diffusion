from .base import BasicAction, from_memory_context, ExecAction
from typing import Dict, Any
import torch

class InputFeederAction(BasicAction):
    @from_memory_context
    def __init__(self, ex_inputs:Dict[str, Any], unet=None):
        super().__init__()
        self.ex_inputs = ex_inputs
        self.unet = unet

    def forward(self, **states):
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(self.ex_inputs)
        return states

class PrepareDiffusionAction(ExecAction):
    dtype_dict = {'fp32':torch.float32, 'amp':torch.float32, 'fp16':torch.float16, 'bf16':torch.bfloat16}

    def __init__(self, dtype='fp32'):
        self.dtype_name = dtype
        self.dtype = self.dtype_dict.get(dtype, torch.float32)

    def forward(self, memory, **states):
        return {'dtype': self.dtype, 'device':memory.pipe.device, **states}