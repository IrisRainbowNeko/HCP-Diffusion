import os
from typing import Dict, List, Tuple, Any

class DataSource:
    def __init__(self, img_root, repeat=1, **kwargs):
        self.img_root = img_root
        self.repeat = repeat

    def get_image_list(self) -> List[Tuple[str, "DataSource"]]:
        raise NotImplementedError()

    def procees_image(self, image):
        raise NotImplementedError()

    def load_image(self, path) -> Dict[str, Any]:
        raise NotImplementedError()

    def get_image_name(self, path: str) -> str:
        img_root, img_name = os.path.split(path)
        return img_name.rsplit('.')[0]

class ComposeDataSource(DataSource):
    def __init__(self, source_dict: Dict[str, DataSource]):
        self.source_dict = source_dict

    def get_image_list(self) -> List[Tuple[str, DataSource]]:
        img_list = []
        for source in self.source_dict.values():
            img_list.extend(source.get_image_list())
        return img_list
