from .base import DataSource
from hcpdiff.data.caption_loader import BaseCaptionLoader, auto_caption_loader
from typing import Union, Any
import os
from hcpdiff.utils.utils import get_file_name, get_file_ext
from hcpdiff.utils.img_size_tool import types_support
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import random
from torchvision.transforms import transforms

default_image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class Text2ImageSource(DataSource):
    def __init__(self, img_root, caption_file, prompt_template, text_transforms, image_transforms=default_image_transforms,
                 bg_color=(255,255,255), repeat=1, **kwargs):
        super(Text2ImageSource, self).__init__(img_root, repeat=repeat)

        self.caption_dict = self.load_captions(caption_file)
        self.prompt_template = self.load_template(prompt_template)
        self.image_transforms = image_transforms
        self.text_transforms = text_transforms
        self.bg_color = tuple(bg_color)

    def load_captions(self, caption_file: Union[str, BaseCaptionLoader]):
        if caption_file is None:
            return {}
        elif isinstance(caption_file, str):
            return auto_caption_loader(caption_file).load()
        else:
            return caption_file.load()

    def load_template(self, template_file):
        with open(template_file, 'r', encoding='utf-8') as f:
            return f.read().strip().split('\n')

    def get_image_list(self) -> List[Tuple[str, DataSource]]:
        imgs = [(os.path.join(self.img_root, x), self) for x in os.listdir(self.img_root) if get_file_ext(x) in types_support]
        return imgs*self.repeat

    def procees_image(self, image):
        return self.image_transforms(image)

    def process_text(self, text_dict):
        return self.text_transforms(text_dict)

    def load_image(self, path) -> Dict[str, Any]:
        image = Image.open(path)
        if image.mode == 'RGBA':
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas
        return {'image': image.convert("RGB")}

    def load_caption(self, img_name) -> str:
        caption_ist = self.caption_dict.get(img_name, None)
        prompt_template = random.choice(self.prompt_template)
        prompt_ist = self.process_text({'prompt':prompt_template, 'caption':caption_ist})['prompt']
        return prompt_ist

class Text2ImageAttMapSource(Text2ImageSource):
    def __init__(self, img_root, caption_file, prompt_template, text_transforms, image_transforms=default_image_transforms, att_mask=None,
                 bg_color=(255, 255, 255), repeat=1, **kwargs):
        super().__init__(img_root, caption_file, prompt_template, image_transforms=image_transforms, text_transforms=text_transforms,
                         bg_color=bg_color, repeat=repeat)

        if att_mask is None:
            self.att_mask = {}
        else:
            self.att_mask = {get_file_name(file):os.path.join(att_mask, file)
                for file in os.listdir(att_mask) if get_file_ext(file) in types_support}

    def get_att_mask(self, img_name):
        if img_name not in self.att_mask:
            return None
        att_mask = Image.open(self.att_mask[img_name]).convert("L")
        np_mask = np.array(att_mask).astype(float)
        np_mask[np_mask<=127+0.1] = (np_mask[np_mask<=127+0.1]/127.)
        np_mask[np_mask>127] = ((np_mask[np_mask>127]-127)/128.)*4+1
        return np_mask

    def load_image(self, path) -> Dict[str, Any]:
        img_root, img_name = os.path.split(path)
        image_dict = super().load_image(path)
        image_dict['att_mask'] = self.get_att_mask(get_file_name(img_name))
        return image_dict