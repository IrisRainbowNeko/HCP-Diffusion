from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SD15_finetuning, cfg_data_SD_ARB

@neko_cfg
def make_cfg():
    return SD15_finetuning(
        base_model='Lykon/DreamShaper',
        train_steps=1000,
        save_step=200,
        dataset=dict(
            dataset1=cfg_data_SD_ARB(
                img_root='imgs/',
                batch_size=4,
                resolution=512*512,
                num_bucket=4,
            )
        )
    )