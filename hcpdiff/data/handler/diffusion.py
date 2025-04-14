from typing import Union, Dict, Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from rainbowneko.data import DataHandler, HandlerChain, LoadImageHandler, ImageHandler

from .text import TemplateFillHandler, TagDropoutHandler, TagEraseHandler, TagShuffleHandler, TokenizeHandler

class LossMapHandler(DataHandler):
    def __init__(self, bucket, vae_scale=8, key_map_in=('loss_map -> image', 'image_size -> image_size'),
                 key_map_out=('image -> loss_map', 'coord -> coord')):
        super().__init__(key_map_in, key_map_out)
        self.vae_scale = vae_scale

        self.handlers = HandlerChain(
            load=LoadImageHandler(mode='L'),
            bucket=bucket.handler,
            image=ImageHandler(transform=T.Compose([
                lambda x:x.resize((x.size[0]//self.vae_scale, x.size[1]//self.vae_scale), Image.BILINEAR),
                T.ToTensor()
            ]), )
        )

    def handle(self, image: Union[Image.Image, str], image_size: np.ndarray[int]):
        data = self.handlers(dict(image=image, image_size=image_size))
        image = data['image']
        image[image<=0.5] *= 2
        image[image>0.5] = (image[image>0.5]-0.5)*4+1
        return self.handlers(dict(**data, image=image))

class DiffusionImageHandler(DataHandler):
    def __init__(self, bucket, key_map_in=('image -> image', 'image_size -> image_size'), key_map_out=('image -> image', 'coord -> coord')):
        super().__init__(key_map_in, key_map_out)

        self.handlers = HandlerChain(
            load=LoadImageHandler(),
            bucket=bucket.handler,
            image=ImageHandler(transform=T.Compose([
                T.ToTensor(),
                T.Normalize([0.5], [0.5])
            ]), )
        )

    def handle(self, image: Image.Image, image_size: np.ndarray[int]):
        if isinstance(image, torch.Tensor):  # cached latents
            return dict(image=image, image_size=image_size)
        else:
            return self.handlers(dict(image=image, image_size=image_size))

class DiffusionTextHandler(DataHandler):
    def __init__(self, encoder_attention_mask=False, erase=0.0, dropout=0.0, shuffle=0.0, word_names={}, tokenize=True,
                 key_map_in=('prompt -> prompt', ), key_map_out=('prompt -> prompt', )):
        super().__init__(key_map_in, key_map_out)

        text_handlers = {}
        if dropout>0:
            text_handlers['dropout'] = TagDropoutHandler(p=dropout)
        if erase>0:
            text_handlers['erase'] = TagEraseHandler(p=erase)
        if shuffle>0:
            text_handlers['shuffle'] = TagShuffleHandler()
        text_handlers['fill'] = TemplateFillHandler(word_names)
        if tokenize:
            text_handlers['tokenize'] = TokenizeHandler(encoder_attention_mask)
        self.handlers = HandlerChain(**text_handlers)

    def handle(self, prompt: Union[str, Dict[str, str]]):
        return self.handlers(dict(prompt=prompt))

class StableDiffusionHandler(DataHandler):
    def __init__(self, bucket, encoder_attention_mask=False, key_map_in=('image -> image', 'image_size -> image_size', 'prompt -> prompt'),
                 key_map_out=('image -> image', 'coord -> coord', 'prompt -> prompt'),
                 erase=0.0, dropout=0.0, shuffle=0.0, word_names={}, tokenize=True):
        super().__init__(key_map_in, key_map_out)

        self.image_handlers = DiffusionImageHandler(bucket)
        self.text_handlers = DiffusionTextHandler(encoder_attention_mask=encoder_attention_mask, erase=erase, dropout=dropout, shuffle=shuffle,
                                                  word_names=word_names, tokenize=tokenize)

    def handle(self, image: Image.Image, image_size: np.ndarray[int], prompt: str):
        return dict(**self.image_handlers(dict(image=image, image_size=image_size)), **self.text_handlers(dict(prompt=prompt)))
    
    def __call__(self, data) -> Dict[str, Any]:
        data_proc = self.handle(**self.key_mapper_in.map_data(data)[1])
        out_data = self.key_mapper_out.map_data(data_proc)[1]
        data = dict(**data)
        data.update(out_data)
        return data
