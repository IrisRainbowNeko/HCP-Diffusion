from typing import Dict, Any

import wandb
from PIL import Image

from .base_logger import BaseLogger


class WanDBLogger(BaseLogger):
    def __init__(self, exp_dir, out_path, enable_log_image=True, project='hcp-diffusion'):
        super().__init__(exp_dir, out_path, enable_log_image)
        if exp_dir is not None:  # exp_dir is only available in local main process
            wandb.init(project=project)
        else:
            self.writer = None
            self.disable()

    def _info(self, info):
        pass

    def _log(self, datas: Dict[str, Any], step: int = 0):
        wandb.log({k: v['data'] for k, v in datas.items()}, step=step)

    def _log_image(self, imgs: Dict[str, Image.Image], step: int = 0):
        wandb.log({next(iter(imgs.keys())): list(imgs.values())}, step=step)
