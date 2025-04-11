from hcpdiff.easy.cfg import SDXL_t2i_lora
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SDXL_t2i_lora(
        pretrained_model='Illustrious-XL-v1.1.safetensors',
        lora_info=[
            ('lora_path', 1.0), # (lora_path, weight)
        ],
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        bs=4,
        width=1024,
        height=1024,
        guidance_scale=7.0
    )