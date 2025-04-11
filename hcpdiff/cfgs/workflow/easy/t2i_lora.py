from hcpdiff.easy.cfg import SD15_t2i_lora
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SD15_t2i_lora(
        pretrained_model='Lykon/DreamShaper',
        lora_info=[
            ('lora_path', 1.0), # (lora_path, weight)
        ],
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        bs=4,
        width=512,
        height=512,
        guidance_scale=7.0
    )