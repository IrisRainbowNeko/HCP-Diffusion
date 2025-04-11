from hcpdiff.easy.cfg import SD15_t2i
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SD15_t2i(
        pretrained_model='Lykon/DreamShaper',
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        bs=4,
        width=512,
        height=512,
        guidance_scale=7.0
    )