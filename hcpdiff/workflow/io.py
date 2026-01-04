import os
from pathlib import Path
import tempfile
import shutil
from functools import partial
from typing import List, Union
from addict import Addict

import torch
from hcpdiff.utils import to_validate_file
from hcpdiff.utils.net_utils import get_dtype
from rainbowneko.ckpt_manager import NekoLoader
from rainbowneko.infer import BasicAction
from rainbowneko.infer import LoadImageAction as Neko_LoadImageAction
from rainbowneko.utils.img_size_tool import types_support
from rainbowneko import _share
from rainbowneko.utils import is_dict
from rainbowneko.loggers import ImageLog, TextFileLog

class BuildModelsAction(BasicAction):
    def __init__(self, model_loader: partial[NekoLoader.load], dtype: str = torch.float32, device='cuda', key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.model_loader = model_loader
        self.dtype = get_dtype(dtype)
        self.device = device

    def forward(self, in_preview=False, model=None, **states):
        if in_preview:
            load_kwargs = dict(dtype=self.dtype, device=self.device, denoiser=model.denoiser, TE=model.TE, vae=model.vae)
            if hasattr(model, 'style_encoder'):
                load_kwargs['style_encoder'] = model.style_encoder
            if hasattr(model, 'token_processor'):
                load_kwargs['token_processor'] = model.token_processor
            if hasattr(model, 'tokenizer'):
                load_kwargs['tokenizer'] = model.tokenizer
            model = self.model_loader(**load_kwargs)
        else:
            model = self.model_loader(dtype=self.dtype, device=self.device)

            # Callback for TokenizerHandler
            if is_dict(model):
                model_wrapper = Addict(model)
            else:
                model_wrapper = model
            for callback in _share.model_callbacks:
                callback(model_wrapper)

        if isinstance(model, dict):
            return model
        else:
            return {'model':model}

class LoadImageAction(Neko_LoadImageAction):
    def __init__(self, image_paths: Union[str, List[str]], image_transforms=None, key_map_in=None, key_map_out=('input.x -> images',)):
        super().__init__(image_paths, image_transforms, key_map_in, key_map_out)

class SaveImageAction(BasicAction):
    def __init__(self, save_root: str, image_type: str = 'png', quality: int = 95, save_cfg=True, save_txt=False, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.save_root = save_root
        self.image_type = image_type
        self.quality = quality
        self.save_cfg = save_cfg
        self.save_txt = save_txt

        os.makedirs(save_root, exist_ok=True)

    def forward(self, images, prompt, negative_prompt, seeds, cfgs=None, parser=None, in_preview=False, preview_step=None, _logs=None, **states):
        save_root = self.save_root
        num_img_exist = max([0]+[int(x.split('-', 1)[0]) for x in os.listdir(save_root) if x.rsplit('.', 1)[-1] in types_support])+1

        if in_preview:
            if _logs is None:
                _logs = {}
            _logs['preview'] = []

        for bid, (p, pn, img) in enumerate(zip(prompt, negative_prompt, images)):
            if in_preview:
                _logs['preview'].append(ImageLog(
                    caption=f'{{step}}-{seeds[bid]}-{p}',
                    image=img
                ))

                if self.save_cfg:
                    # Create temporary directory to save config files
                    temp_dir = Path(tempfile.mkdtemp())
                    try:
                        cfgs.seed = seeds[bid]
                        config_filename = f"{seeds[bid]}-info"
                        parser.save_configs(cfgs, temp_dir/config_filename)

                        # Add all saved config files to _logs
                        for file in temp_dir.rglob("*"):
                            if file.is_file():
                                rel_path = file.relative_to(temp_dir)
                                _logs[f"config/{rel_path.parent}"] = TextFileLog(
                                    text=file.read_text(encoding='utf-8'),
                                    file_name='{step}-'+rel_path.name
                                )
                    finally:
                        # Clean up temporary directory
                        shutil.rmtree(temp_dir, ignore_errors=True)

                if self.save_txt:
                    _logs["txt"] = TextFileLog(
                        text=p,
                        file_name=f"{{step}}-{seeds[bid]}.txt"
                    )
            else:
                img_path = os.path.join(save_root, f"{preview_step or num_img_exist}-{seeds[bid]}-{to_validate_file(p)}.{self.image_type}")
                img.save(img_path, quality=self.quality)

                if self.save_cfg:
                    cfgs.seed = seeds[bid]
                    parser.save_configs(cfgs, os.path.join(save_root, f"{preview_step or num_img_exist}-{seeds[bid]}-info"))

                if self.save_txt:
                    txt_path = os.path.join(save_root, f"{preview_step or num_img_exist}-{seeds[bid]}-{to_validate_file(prompt[0])}.txt")
                    with open(txt_path, 'w') as f:
                        f.write(p)
                num_img_exist += 1
        return {'_logs':_logs}