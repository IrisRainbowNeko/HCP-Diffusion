from typing import List, Union

import torch
from PIL import Image
from hcpdiff.data.handler import ControlNetHandler
from rainbowneko.infer import BasicAction
from torch import nn

class LatentResizeAction(BasicAction):
    def __init__(self, width=1024, height=1024, mode='bicubic', antialias=True, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.size = (height//8, width//8)
        self.mode = mode
        self.antialias = antialias

    def forward(self, latents, **states):
        latents_dtype = latents.dtype
        latents = nn.functional.interpolate(latents.to(dtype=torch.float32), size=self.size, mode=self.mode)
        latents = latents.to(dtype=latents_dtype)
        return {'latents':latents}

class ImageResizeAction(BasicAction):
    # resample name to Image.xxx
    mode_map = {'nearest':Image.NEAREST, 'bilinear':Image.BILINEAR, 'bicubic':Image.BICUBIC, 'lanczos':Image.LANCZOS, 'box':Image.BOX,
        'hamming':Image.HAMMING, 'antialias':Image.LANCZOS}

    def __init__(self, width=1024, height=1024, mode='bicubic', key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.size = (width, height)
        self.mode = self.mode_map[mode]

    def forward(self, images: List[Image.Image], **states):
        images = [image.resize(self.size, resample=self.mode) for image in images]
        return {'images':images}

class FeedtoCNetAction(BasicAction):
    def __init__(self, width=None, height=None, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.size = (width, height)
        self.cnet_handler = ControlNetHandler()

    def forward(self, images: Union[List[Image.Image], Image.Image], device='cuda', dtype=None, bs=None, latents=None, **states):
        if bs is None:
            if 'prompt' in states:
                bs = len(states['prompt'])

        if latents is not None:
            width, height = latents.shape[3]*8, latents.shape[2]*8
        else:
            width, height = self.size

        images = self.cnet_handler.handle(images).to(device, dtype=dtype).expand(bs*2, 3, width, height)
        return {'ex_inputs':{'cond':images}}
