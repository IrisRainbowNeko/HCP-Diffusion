from copy import copy
from typing import Union

from rainbowneko.data.label_loader import BaseLabelLoader, auto_label_loader

from .text2img import Text2ImageLossMapSource

class T2IFolderClassSource(Text2ImageLossMapSource):
    def _load_label_data(self, label_file: Union[str, BaseLabelLoader]):
        ''' {class_name/image.ext: label} '''
        if label_file is None:
            return {}
        elif isinstance(label_file, str):
            captions = {}
            caption_loader = auto_label_loader(label_file)
            for class_folder in caption_loader.path.iterdir():
                caption_loader_class = copy(caption_loader)
                caption_loader_class.path = class_folder
                captions_class = {f'{class_folder.name}/{name}':caption for name, caption in caption_loader_class.load().item()}
                captions.update(captions_class)
            return captions
        else:
            return label_file.load()
