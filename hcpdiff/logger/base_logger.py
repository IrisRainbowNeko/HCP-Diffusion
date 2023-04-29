from typing import Dict, Any
from PIL import Image

class BaseLogger:
    def __init__(self, exp_dir, out_path, enable_log_image=True):
        self.exp_dir=exp_dir
        self.out_path=out_path
        self.enable_log_image=enable_log_image

    def enable(self):
        raise NotImplementedError()

    def disable(self):
        raise NotImplementedError()

    def info(self, info):
        raise NotImplementedError()

    def log(self, datas:Dict[str, Any], step:int=0):
        raise NotImplementedError()

    def log_image(self, imgs:Dict[str, Image], step:int=0):
        raise NotImplementedError()

