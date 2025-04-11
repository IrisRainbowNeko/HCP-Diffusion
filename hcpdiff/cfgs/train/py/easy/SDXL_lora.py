from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SDXL_lora_train, cfg_data_SD_ARB

@neko_cfg
def make_cfg():
    return SDXL_lora_train(
        base_model='stabilityai/stable-diffusion-xl-base-1.0',
        train_steps=1000,
        save_step=200,
        rank=8,
        low_vram=True,
        dataset=dict(
            dataset1=cfg_data_SD_ARB(
                img_root='imgs/',
                batch_size=4,
                trigger_word='paimeng',
                resolution=1024*1024,
                num_bucket=4,
            )
        )
    )