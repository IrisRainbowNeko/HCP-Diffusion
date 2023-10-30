import os
from typing import Dict, Any

from PIL import Image
from torchvision import transforms

from .text2img import Text2ImageAttMapSource, default_image_transforms

class Text2ImageCondSource(Text2ImageAttMapSource):
    def __init__(self, img_root, caption_file, prompt_template, text_transforms, image_transforms=default_image_transforms,
                bg_color=(255, 255, 255), repeat=1, cond_dir=None, **kwargs):
        super().__init__(img_root, caption_file, prompt_template, image_transforms=image_transforms, text_transforms=text_transforms,
                         bg_color=bg_color, repeat=repeat)
        self.cond_transform = transforms.ToTensor()
        self.cond_dir = cond_dir

    def load_image(self, path) -> Dict[str, Any]:
        img_root, img_name = os.path.split(path)
        image_dict = super().load_image(path)
        cond_path = os.path.join(self.cond_dir, img_name)
        image_dict['cond'] = Image.open(cond_path).convert("RGB")
        return image_dict
