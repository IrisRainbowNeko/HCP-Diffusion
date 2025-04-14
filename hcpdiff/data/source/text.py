from rainbowneko.data import UnLabelSource, DataSource
from rainbowneko.data.label_loader import BaseLabelLoader, auto_label_loader
from typing import Union, Dict, Any
import random

class TextSource(DataSource):
    def __init__(self, label_file, prompt_template=None, repeat=1, **kwargs):
        super().__init__(repeat=repeat)
        self.label_file = label_file
        self.label_dict = self._load_label_data(label_file)
        self.img_ids = self._load_img_ids(self.label_dict)
        self.prompt_template = self.load_template(prompt_template)

    def _load_img_ids(self, label_dict):
        return list(label_dict.keys()) * self.repeat

    def _load_label_data(self, label_file: Union[str, BaseLabelLoader]):
        if label_file is None:
            return {}
        elif isinstance(label_file, str):
            return auto_label_loader(label_file).load()
        else:
            return label_file.load()

    def load_template(self, template_file):
        if template_file is None:
            return ['{caption}']
        else:
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read().strip().split('\n')

    def __getitem__(self, index) -> Dict[str, Any]:
        img_name = self.img_ids[index]
        return {
            'id':img_name,
            'prompt':{
                'template':random.choice(self.prompt_template),
                'caption':self.label_dict[img_name],
            }
        }