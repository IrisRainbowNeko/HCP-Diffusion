from .disk_interface import DiskInterface
from loguru import logger

class WebUIInterface(DiskInterface):

    def __init__(self, save_root, image_type='png', quality=95, show_steps=1, show_inter=False):
        super(WebUIInterface, self).__init__(save_root, image_type, quality, show_steps)
        self.show_inter = show_inter
        self.need_inter_imgs = self.need_inter_imgs and show_inter

    def on_inter_step(self, i, num_steps, t, latents, images):
        if self.show_inter:
            super(WebUIInterface, self).on_inter_step(i, num_steps, t, latents, images)
        logger.info(f'\nthis progress steps: {i}/{num_steps}')

    def on_save_one(self, num_img_exist, img_path):
        logger.info(f'this images output path: {img_path}')
