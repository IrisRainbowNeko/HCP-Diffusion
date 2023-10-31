import os
import re

from imgutils.sd import SDMetaData
from omegaconf import OmegaConf

from hcpdiff.utils.img_size_tool import types_support
from hcpdiff.utils.utils import to_validate_file
from .base_interface import BaseInterface

def _read_scheduler_name(cfg) -> str:
    _suffix_ = cfg._target_.split('.')[-1]
    if _suffix_ == 'DPMSolverMultistepScheduler':
        if cfg.get('use_karras_sigmas'):
            if cfg.get('algorithm_type') == 'sde-dpmsolver++':
                return 'DPM++ 2M SDE Karras'
            else:
                return 'DPM++ 2M Karras'
        else:
            if cfg.get('algorithm_type') == 'sde-dpmsolver++':
                return 'DPM++ 2M SDE'
            else:
                return 'DPM++ 2M'
    elif _suffix_ == 'DPMSolverSinglestepScheduler':
        if cfg.get('use_karras_sigmas'):
            return 'DPM++ SDE Karras'
        else:
            return 'DPM++ SDE'
    elif _suffix_ == 'KDPM2DiscreteScheduler':
        if cfg.get('use_karras_sigmas'):
            return 'DPM2 Karras'
        else:
            return 'DPM2'
    elif _suffix_ == 'KDPM2AncestralDiscreteScheduler':
        if cfg.get('use_karras_sigmas'):
            return 'DPM2 a Karras'
        else:
            return 'DPM2 a'
    elif _suffix_ == 'EulerDiscreteScheduler':
        return 'Euler'
    elif _suffix_ == 'EulerAncestralDiscreteScheduler':
        return 'Euler a'
    elif _suffix_ == 'HeunDiscreteScheduler':
        return 'Heun'
    elif _suffix_ == 'LMSDiscreteScheduler':
        if cfg.get('use_karras_sigmas'):
            return 'LMS Karras'
        else:
            return 'LMS'

    word = _suffix_
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
    word = word.replace('-', '_')
    words = [w for w in re.split(r'_+', word) if w and w.lower() != 'scheduler']
    if cfg.get('use_karras_sigmas') and words[-1].lower() != 'karras':
        words.append('Karras')
    return ' '.join(words)

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
            if self.save_cfg and cfgs_raw is not None:
                with open(os.path.join(self.save_root, f"{num_img_exist}-{seeds[bid]}-info.yaml"), 'w', encoding='utf-8') as f:
                    cfgs_raw.seed = seeds[bid]
                    f.write(OmegaConf.to_yaml(cfgs_raw))

                sd_metadata = SDMetaData(
                    prompt=cfgs_raw.prompt,
                    neg_prompt=cfgs_raw.neg_prompt,
                    parameters={
                        'CFG scale':cfgs_raw.infer_args.guidance_scale,
                        'Clip skip':cfgs_raw.clip_skip+1,
                        'Model':cfgs_raw.pretrained_model,
                        # 'Model hash':'54ef3e3610',
                        'Sampler':_read_scheduler_name(cfgs_raw.new_components.scheduler),
                        'Seed':cfgs_raw.seed,
                        'Size':(cfgs_raw.infer_args.width, cfgs_raw.infer_args.height),
                        'Steps':cfgs_raw.infer_args.num_inference_steps,
                    }
                )
            else:
                sd_metadata = None

            img_path = os.path.join(self.save_root, f"{num_img_exist}-{seeds[bid]}-{to_validate_file(prompt[0])}.{self.image_type}")
            if sd_metadata is not None:
                img.save(img_path, quality=self.quality, pnginfo=sd_metadata.pnginfo)
            else:
                img.save(img_path, quality=self.quality)
            self.on_save_one(num_img_exist, img_path)

            if self.need_inter_imgs:
                inter = self.inter_imgs[bid]
                inter[0].save(os.path.join(self.save_root, f'{num_img_exist}-{seeds[bid]}-steps.webp'), "webp", save_all=True,
                              append_images=inter[1:], duration=100)
            num_img_exist += 1
