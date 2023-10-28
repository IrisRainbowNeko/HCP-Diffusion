import os
from typing import List, Tuple, Union
from hcpdiff.utils.utils import get_file_name, get_file_ext
from hcpdiff.utils.img_size_tool import types_support
from .text2img import Text2ImageAttMapSource
from hcpdiff.data.caption_loader import BaseCaptionLoader, auto_caption_loader
from copy import copy

class T2IFolderClassSource(Text2ImageAttMapSource):

    def get_image_list(self) -> List[Tuple[str, "T2IFolderClassSource"]]:
        sub_folders = [os.path.join(self.img_root, x) for x in os.listdir(self.img_root)]
        class_imgs = []
        for class_folder in sub_folders:
            class_name = os.path.basename(class_folder)
            imgs = [(os.path.join(class_folder, x), self) for x in os.listdir(class_folder) if get_file_ext(x) in types_support]
            class_imgs.extend(imgs*self.repeat[class_name])
        return class_imgs

    def load_captions(self, caption_file: Union[str, BaseCaptionLoader]):
        if caption_file is None:
            return {}
        elif isinstance(caption_file, str):
            captions = {}
            caption_loader = auto_caption_loader(caption_file)
            for class_name in os.listdir(caption_loader.path):
                class_folder = os.path.join(caption_loader.path, class_name)
                caption_loader_class = copy(caption_loader)
                caption_loader_class.path = class_folder
                captions_class = {f'{class_name}/{name}':caption for name, caption in caption_loader_class.load().item()}
                captions.update(captions_class)
            return captions
        else:
            return caption_file.load()

    def get_image_name(self, path: str) -> str:
        img_root, img_name = os.path.split(path)
        img_name = img_name.rsplit('.')[0]
        img_root, class_name = os.path.split(img_root)
        return f'{class_name}/{img_name}'
