from typing import Dict, Any

from PIL import Image
import wandb

from .cli_logger import CLILogger


class WanDBLogger(CLILogger):
    def __init__(self, exp_dir, out_path, enable_log_image=True, project='hcp-diffusion'):
        super().__init__(exp_dir, out_path, enable_log_image)
        self.enable_log = True
        if exp_dir is not None:  # exp_dir is only available in local main process
            wandb.init(project=project)
        else:
            self.writer = None
            self.disable()

    def enable(self):
        super(WanDBLogger, self).enable()
        self.enable_log = True

    def disable(self):
        super(WanDBLogger, self).disable()
        self.enable_log = False

    def log(self, datas: Dict[str, Any], step: int = 0):
        if self.enable_log:
            wandb.log({k:v['data'] for k,v in datas.items()}, step=step)

    def log_image(self, imgs: Dict[str, Image], step: int = 0):
        if self.enable_log and self.enable_log_image:
            wandb.log({next(iter(imgs.keys())): list(imgs.values())}, step=step)
