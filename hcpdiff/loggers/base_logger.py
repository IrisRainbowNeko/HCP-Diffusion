from typing import Dict, Any, List

from PIL import Image

from .preview import ImagePreviewer

class BaseLogger:
    def __init__(self, exp_dir, out_path, enable_log_image=False, log_step=10, image_log_step=200):
        self.exp_dir = exp_dir
        self.out_path = out_path
        self.enable_log_image = enable_log_image
        self.log_step = log_step
        self.image_log_step = image_log_step
        self.enable_log = True
        self.previewer_list: List[ImagePreviewer] = []

    def enable(self):
        self.enable_log = True

    def disable(self):
        self.enable_log = False

    def add_previewer(self, previewer: ImagePreviewer):
        self.previewer_list.append(previewer)

    def info(self, info):
        if self.enable_log:
            self._info(info)

    def _info(self, info):
        raise NotImplementedError()

    def log(self, datas: Dict[str, Any], step: int = 0):
        if self.enable_log and step%self.log_step == 0:
            self._log(datas, step)

    def _log(self, datas: Dict[str, Any], step: int = 0):
        raise NotImplementedError()

    def log_image(self, imgs: Dict[str, Image.Image], step: int = 0):
        if self.enable_log and self.enable_log_image and step%self.image_log_step == 0:
            self._log_image(imgs, step)

    def _log_image(self, imgs: Dict[str, Image.Image], step: int = 0):
        raise NotImplementedError()

class LoggerGroup:
    def __init__(self, logger_list: List[BaseLogger]):
        self.logger_list = logger_list

    def enable(self):
        for logger in self.logger_list:
            logger.enable()

    def disable(self):
        for logger in self.logger_list:
            logger.disable()

    def add_previewer(self, previewer):
        for logger in self.logger_list:
            logger.add_previewer(previewer)

    def info(self, info):
        for logger in self.logger_list:
            logger.info(info)

    def log(self, datas: Dict[str, Any], step: int = 0):
        for logger in self.logger_list:
            logger.log(datas, step)

    def log_image(self, imgs: Dict[str, Image.Image], step: int = 0):
        for logger in self.logger_list:
            logger.log_image(imgs, step)

    def __len__(self):
        return len(self.logger_list)
