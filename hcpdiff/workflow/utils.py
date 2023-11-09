from .base import BasicAction, from_memory_context
from torch import nn

class LatentResizeAction(BasicAction):
    @from_memory_context
    def __init__(self, scale_factor=2, mode='bicubic', antialias=True):
        self.scale_factor = scale_factor
        self.mode = mode
        self.antialias = antialias

    def forward(self, latents, **states):
        latents = nn.functional.interpolate(latents, scale_factor=self.scale_factor, mode=self.mode, antialias=self.antialias)
        return {'latents':latents, **states}