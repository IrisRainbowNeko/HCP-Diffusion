from typing import Dict, Any

import os
import wandb
from PIL import Image

from .base_logger import BaseLogger


class WanDBLogger(BaseLogger):
    def __init__(self, exp_dir, out_path=None, enable_log_image=True, project='hcp-diffusion', log_step=10):
        super().__init__(exp_dir, out_path, enable_log_image, log_step)
        if exp_dir is not None:  # exp_dir is only available in local main process
            wandb.init(project=project, name=os.path.basename(exp_dir))
            wandb.save(os.path.join(exp_dir, 'cfg.yaml'), base_path=exp_dir)
        else:
            self.writer = None
            self.disable()

    def _info(self, info):
        pass

    def _log(self, datas: Dict[str, Any], step: int = 0):
        log_dict = {'step': step}
        for k, v in datas.items():
            if len(v['data']) == 1:
                log_dict[k] = v['data'][0]
        wandb.log(log_dict)

    def _log_image(self, imgs: Dict[str, Image.Image], step: int = 0):
        wandb.log({next(iter(imgs.keys())): list(imgs.values())}, step=step)
