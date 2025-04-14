from hcpdiff.easy.cfg import SD15_t2i_lora
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return dict(
        pretrained_model='Lykon/DreamShaper',
        lora_path='',
        lora_weight=1.0,
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        bs=4,
        width=512,
        height=512,
        guidance_scale=7.0

        **SD15_t2i_lora(
            pretrained_model='${pretrained_model}',
            lora_info=[
                ('${lora_path}', '${lora_weight}'),
            ],
            prompt='${prompt}',
            bs='${bs}',
            width='${width}',
            height='${height}',
            guidance_scale='${guidance_scale}',
        )
    )