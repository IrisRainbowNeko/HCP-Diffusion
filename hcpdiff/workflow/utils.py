import torch

from .base import BasicAction, from_memory_context, feedback_input
from torch import nn
from PIL import Image
from typing import List
from hcpdiff.data.data_processor import ControlNetProcessor
from hcpdiff.utils import get_dtype

class LatentResizeAction(BasicAction):
    @from_memory_context
    def __init__(self, width=1024, height=1024, mode='bicubic', antialias=True):
        self.size = (height//8, width//8)
        self.mode = mode
        self.antialias = antialias

    @feedback_input
    def forward(self, latents, **states):
        latents_dtype = latents.dtype
        latents = nn.functional.interpolate(latents.to(dtype=torch.float32), size=self.size, mode=self.mode)
        latents = latents.to(dtype=latents_dtype)
        return {'latents':latents}

class ImageResizeAction(BasicAction):
    # resample name to Image.xxx
    mode_map = {'nearest':Image.NEAREST, 'bilinear':Image.BILINEAR, 'bicubic':Image.BICUBIC, 'lanczos':Image.LANCZOS, 'box':Image.BOX,
        'hamming':Image.HAMMING, 'antialias':Image.ANTIALIAS}

    @from_memory_context
    def __init__(self, width=1024, height=1024, mode='bicubic'):
        self.size = (width, height)
        self.mode = self.mode_map[mode]

    @feedback_input
    def forward(self, images: List[Image.Image], **states):
        images = [image.resize(self.size, resample=self.mode) for image in images]
        return {'images':images}

class FeedtoCNetAction(BasicAction):
    @from_memory_context
    def __init__(self, width=None, height=None):
        self.size = (width, height)

    @feedback_input
    def forward(self, images: List[Image.Image], device='cuda', dtype=None, bs=None, latents=None, **states):
        if bs is None:
            if 'prompt' in states:
                bs = len(states['prompt'])

        if latents is not None:
            width, height = latents.shape[3]*8, latents.shape[2]*8
        else:
            width, height = self.size

        images = ControlNetProcessor.prepare_cond_image(images, width, height, bs*2, device).to(dtype=get_dtype(dtype))
        return {'_ex_input':{'cond':images}}
