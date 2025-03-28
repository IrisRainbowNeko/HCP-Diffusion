from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SDXL_lora_train, cfg_data_SD_ARB, SD15_t2i
from hcpdiff.evaluate import HCPPreviewer

@neko_cfg
def make_cfg():
    return dict(
        **SDXL_lora_train(
            base_model='Lykon/DreamShaper',
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
        ),
        evaluator=HCPPreviewer(_partial_=True,
            interval=100,
            workflow=SD15_t2i,
        ),
    )