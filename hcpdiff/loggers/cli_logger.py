import os
from typing import Dict, Any

from PIL import Image
from loguru import logger

from .base_logger import BaseLogger


class CLILogger(BaseLogger):
    def __init__(self, exp_dir, out_path, enable_log_image=True, log_step=10, img_ext='png', img_quality=95):
        super().__init__(exp_dir, out_path, enable_log_image, log_step)
        if exp_dir is not None:  # exp_dir is only available in local main process
            logger.add(os.path.join(exp_dir, out_path))
            self.img_log_dir = os.path.join(exp_dir, os.path.basename(out_path), 'imgs/')
            self.img_ext = img_ext
            self.img_quality = img_quality
        else:
            self.disable()

    def enable(self):
        super(CLILogger, self).enable()
        logger.enable("__main__")

    def disable(self):
        super(CLILogger, self).disable()
        logger.disable("__main__")

    def _info(self, info):
        logger.info(info)

    def _log(self, datas: Dict[str, Any], step:int=0):
        logger.info(', '.join([f"{k} = {v['format'].format(*v['data'])}" for k, v in datas.items()]))

    def _log_image(self, imgs: Dict[str, Image.Image], step:int=0):
        for name, img in imgs.items():
            img.save(f'{name}.{self.img_ext}', quality=self.img_quality)
