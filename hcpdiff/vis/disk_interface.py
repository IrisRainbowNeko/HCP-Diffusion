import os

from hcpdiff.utils.img_size_tool import types_support
from hcpdiff.utils.utils import to_validate_file
from omegaconf import OmegaConf

from .base_interface import BaseInterface

class DiskInterface(BaseInterface):
    def __init__(self, save_root, save_cfg=True, image_type='png', quality=95, show_steps=0):
        super(DiskInterface, self).__init__(show_steps=show_steps)
        os.makedirs(save_root, exist_ok=True)
        self.save_root = save_root
        self.save_cfg = save_cfg
        self.image_type = image_type
        self.quality = quality

        self.inter_imgs = []
        if show_steps>0:
            self.need_inter_imgs = True

    def on_inter_step(self, i, num_steps, t, latents, images):
        if len(self.inter_imgs) == 0:
            for _ in range(len(images)):
                self.inter_imgs.append([])
        for u, img in enumerate(images):
            self.inter_imgs[u].append(img)

    def on_save_one(self, num_img_exist, img_path):
        pass

    def on_infer_finish(self, images, prompt, negative_prompt, cfgs_raw=None, seeds=None):
        num_img_exist = max([0]+[int(x.split('-', 1)[0]) for x in os.listdir(self.save_root) if x.rsplit('.', 1)[-1] in types_support])+1

        for bid, (p, pn, img) in enumerate(zip(prompt, negative_prompt, images)):
            img_path = os.path.join(self.save_root, f"{num_img_exist}-{seeds[bid]}-{to_validate_file(prompt[0])}.{self.image_type}")
            img.save(img_path, quality=self.quality)
            self.on_save_one(num_img_exist, img_path)

            if self.save_cfg and cfgs_raw is not None:
                with open(os.path.join(self.save_root, f"{num_img_exist}-{seeds[bid]}-info.yaml"), 'w', encoding='utf-8') as f:
                    cfgs_raw.seed = seeds[bid]
                    f.write(OmegaConf.to_yaml(cfgs_raw))
            if self.need_inter_imgs:
                inter = self.inter_imgs[bid]
                inter[0].save(os.path.join(self.save_root, f'{num_img_exist}-{seeds[bid]}-steps.webp'), "webp", save_all=True,
                              append_images=inter[1:], duration=100)
            num_img_exist += 1
