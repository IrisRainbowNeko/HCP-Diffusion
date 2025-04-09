import os
from functools import partial
from typing import List, Union

import torch
from hcpdiff.utils import to_validate_file
from hcpdiff.utils.net_utils import get_dtype
from rainbowneko.ckpt_manager import NekoLoader
from rainbowneko.infer import BasicAction
from rainbowneko.infer import LoadImageAction as Neko_LoadImageAction
from rainbowneko.utils.img_size_tool import types_support

class BuildModelsAction(BasicAction):
    def __init__(self, model_loader: partial[NekoLoader.load], dtype: str=torch.float32, device='cuda', key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.model_loader = model_loader
        self.dtype = get_dtype(dtype)
        self.device = device

    def forward(self, in_preview=False, model=None, **states):
        if in_preview:
            model = self.model_loader(dtype=self.dtype, device=self.device, denoiser=model.denoiser, TE=model.TE, vae=model.vae)
        else:
            model = self.model_loader(dtype=self.dtype, device=self.device)

        if isinstance(model, dict):
            return model
        else:
            return {'model':model}

class LoadImageAction(Neko_LoadImageAction):
    def __init__(self, image_paths: Union[str, List[str]], image_transforms=None, key_map_in=None, key_map_out=('input.x -> images',)):
        super().__init__(image_paths, image_transforms, key_map_in, key_map_out)

class SaveImageAction(BasicAction):
    def __init__(self, save_root: str, image_type: str = 'png', quality: int = 95, save_cfg=True, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.save_root = save_root
        self.image_type = image_type
        self.quality = quality
        self.save_cfg = save_cfg

        os.makedirs(save_root, exist_ok=True)

    def forward(self, images, prompt, negative_prompt, seeds, cfgs=None, parser=None, preview_root=None, preview_step=None, **states):
        save_root = preview_root or self.save_root
        num_img_exist = max([0]+[int(x.split('-', 1)[0]) for x in os.listdir(save_root) if x.rsplit('.', 1)[-1] in types_support])+1

        for bid, (p, pn, img) in enumerate(zip(prompt, negative_prompt, images)):
            img_path = os.path.join(save_root, f"{preview_step or num_img_exist}-{seeds[bid]}-{to_validate_file(prompt[0])}.{self.image_type}")
            img.save(img_path, quality=self.quality)
            num_img_exist += 1

            if self.save_cfg:
                cfgs.seed = seeds[bid]
                parser.save_configs(cfgs, os.path.join(save_root, f"{preview_step or num_img_exist}-{seeds[bid]}-info"))
