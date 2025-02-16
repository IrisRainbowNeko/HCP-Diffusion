import os
from typing import Dict, Any

from .text2img import Text2ImageSource

class Text2ImageCondSource(Text2ImageSource):
    def __init__(self, img_root, caption_file, prompt_template, repeat=1, cond_dir=None, **kwargs):
        super().__init__(img_root, caption_file, prompt_template, repeat=repeat)
        self.cond_dir = cond_dir

    def __getitem__(self, index) -> Dict[str, Any]:
        data = super().__getitem__(index)
        img_name = self.img_ids[index]
        cond_path = os.path.join(self.cond_dir, img_name)
        data['cond'] = cond_path
        return data
