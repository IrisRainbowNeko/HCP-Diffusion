import os
import random
from pathlib import Path
from typing import Any
from typing import Dict

from rainbowneko.data import ImageLabelSource
from rainbowneko.utils.utils import is_image_file
from torchvision.transforms import transforms

default_image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class Text2ImageSource(ImageLabelSource):
    def __init__(self, img_root, label_file, prompt_template, repeat=1, **kwargs):
        super().__init__(img_root, label_file, repeat=repeat)

        self.prompt_template = self.load_template(prompt_template)

    def load_template(self, template_file):
        with open(template_file, 'r', encoding='utf-8') as f:
            return f.read().strip().split('\n')

    def __getitem__(self, index) -> Dict[str, Any]:
        img_name = self.img_ids[index]
        path = self.img_root / img_name

        return {
            'id':img_name,
            'image':path,
            'prompt':{
                'template':random.choice(self.prompt_template),
                'caption':self.label_dict.get(img_name, None),
            }
        }

class Text2ImageLossMapSource(Text2ImageSource):
    def __init__(self, img_root, caption_file, prompt_template, loss_map=None, repeat=1, **kwargs):
        super().__init__(img_root, caption_file, prompt_template, repeat=repeat)

        if loss_map is None:
            self.loss_map = {}
        else:
            loss_map = Path(loss_map)
            self.loss_map = {file.stem:loss_map/file for file in loss_map.iterdir() if is_image_file(file)}

    def __getitem__(self, index) -> Dict[str, Any]:
        data = super().__getitem__(index)
        img_name = self.img_ids[index]
        data['loss_map'] = self.loss_map[Path(img_name).stem]
        return data
