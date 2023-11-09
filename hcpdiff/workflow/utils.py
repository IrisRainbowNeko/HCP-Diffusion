from .base import BasicAction, from_memory_context
from torch import nn
from PIL import Image
from typing import List

class LatentResizeAction(BasicAction):
    @from_memory_context
    def __init__(self, scale_factor=2, mode='bicubic', antialias=True):
        self.scale_factor = scale_factor
        self.mode = mode
        self.antialias = antialias

    def forward(self, latents, **states):
        latents = nn.functional.interpolate(latents, scale_factor=self.scale_factor, mode=self.mode)
        return {**states, 'latents':latents}

class ImageResizeAction(BasicAction):
    # resample name to Image.xxx
    mode_map = {'nearest':Image.NEAREST, 'bilinear':Image.BILINEAR, 'bicubic':Image.BICUBIC, 'lanczos':Image.LANCZOS, 'box':Image.BOX,
        'hamming':Image.HAMMING, 'antialias':Image.ANTIALIAS}

    @from_memory_context
    def __init__(self, scale_factor=2, mode='bicubic'):
        self.scale_factor = scale_factor
        self.mode = self.mode_map[mode]

    def forward(self, images:List[Image.Image], **states):
        images = [image.resize((int(image.width*self.scale_factor), int(image.height*self.scale_factor)), resample=self.mode) for image in images]
        return {**states, 'images':images}