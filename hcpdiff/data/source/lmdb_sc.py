from .text2img import Text2ImageSource, default_image_transforms
from .base import DataSource
from typing import List, Tuple, Dict, Any
import lmdb
from PIL import Image
from io import BytesIO

class LMDBT2ISource(Text2ImageSource):
    def __init__(self, img_root, caption_file, prompt_template, text_transforms, image_transforms=default_image_transforms,
                 bg_color=(255,255,255), repeat=1, buffer_size=1000, **kwargs):
        super().__init__(img_root, caption_file, prompt_template, text_transforms, image_transforms, bg_color, repeat, **kwargs)

        self.env = lmdb.open(img_root, readonly=True, lock=False)  # 打开 LMDB 数据库

    def get_image_list(self) -> List[Tuple[str, DataSource]]:
        imgs = list(self.caption_dict.keys())
        return imgs*self.repeat

    def load_image(self, name:str) -> Dict[str, Any]:
        with self.env.begin(write=False) as txn:
            img_data = txn.get(name.encode())  # 从 LMDB 获取图像数据
        image = Image.open(BytesIO(img_data))
        if image.mode == 'RGBA':
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas
        return {'image': image.convert("RGB")}