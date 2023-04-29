import os
from typing import Dict, Any

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from .cli_logger import CLILogger


class TBLogger(CLILogger):
    def __init__(self, exp_dir, out_path, enable_log_image=True):
        super().__init__(exp_dir, out_path, enable_log_image)
        self.enable_log = True
        if exp_dir is not None:  # exp_dir is only available in local main process
            self.writer = SummaryWriter(os.path.join(exp_dir, out_path))
        else:
            self.writer = None
            self.disable()

    def enable(self):
        super(TBLogger, self).enable()
        self.enable_log = True

    def disable(self):
        super(TBLogger, self).disable()
        self.enable_log = False

    def log(self, datas: Dict[str, Any], step: int = 0):
        if self.enable_log:
            for k, v in datas.items():
                if len(v['data']) == 1:
                    self.writer.add_scalar(k, v['data'][0], global_step=step)

    def log_image(self, imgs: Dict[str, Image], step: int = 0):
        if self.enable_log and self.enable_log_image:
            for name, img in imgs.items():
                self.writer.add_image(f'img/{name}', np.array(img), dataformats='HWC', global_step=step)
