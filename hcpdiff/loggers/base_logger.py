import os
from typing import Dict, Any, List

from PIL import Image
from omegaconf import OmegaConf

from hcpdiff.utils.utils import to_validate_file
from .preview import ImagePreviewer

class BaseLogger:
    def __init__(self, exp_dir, out_path, enable_log_image=True, log_step=10, image_log_step=200):
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

    def log_preview(self, step: int = 0):
        if self.enable_log and self.enable_log_image and step%self.image_log_step == 0:
            for previewer in self.previewer_list:
                prompt_all, negative_prompt_all, seeds_all, images_all, cfgs_raw = previewer.preview()
                imgs = {os.path.join(previewer.preview_dir, f'{step}-{seed}-{to_validate_file(prompt)}'):img for prompt, seed, img
                    in zip(prompt_all, seeds_all, images_all)}
                self.log_image(imgs, step)

                if cfgs_raw is not None:
                    for seed in seeds_all:
                        with open(os.path.join(previewer.preview_dir, f"{step}-{seed}-info.yaml"), 'w', encoding='utf-8') as f:
                            cfgs_raw.seed = seed
                            f.write(OmegaConf.to_yaml(cfgs_raw))

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

    def log_preview(self, step: int = 0):
        for logger in self.logger_list:
            logger.log_preview(step)

    def __len__(self):
        return len(self.logger_list)
