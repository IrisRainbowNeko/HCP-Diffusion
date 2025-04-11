from hcpdiff.easy.cfg import SDXL_t2i
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SDXL_t2i(
        pretrained_model='Illustrious-XL-v1.1.safetensors',
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        bs=1,
        width=1024,
        height=1024,
        guidance_scale=7.0
    )