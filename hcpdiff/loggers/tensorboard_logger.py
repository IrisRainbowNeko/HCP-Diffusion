import os
from typing import Dict, Any

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from .base_logger import BaseLogger


class TBLogger(BaseLogger):
    def __init__(self, exp_dir, out_path, enable_log_image=False, log_step=10, image_log_step=200):
        super().__init__(exp_dir, out_path, enable_log_image, log_step, image_log_step)
        if exp_dir is not None:  # exp_dir is only available in local main process
            self.writer = SummaryWriter(os.path.join(exp_dir, out_path))
        else:
            self.writer = None
            self.disable()

    def _info(self, info):
        pass

    def _log(self, datas: Dict[str, Any], step: int = 0):
        for k, v in datas.items():
            if len(v['data']) == 1:
                self.writer.add_scalar(k, v['data'][0], global_step=step)

    def _log_image(self, imgs: Dict[str, Image.Image], step: int = 0):
        for name, img in imgs.items():
            self.writer.add_image(f'img/{name}', np.array(img), dataformats='HWC', global_step=step)
