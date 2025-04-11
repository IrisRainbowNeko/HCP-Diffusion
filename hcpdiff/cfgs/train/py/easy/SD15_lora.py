from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SD15_lora_train, cfg_data_SD_ARB

@neko_cfg
def make_cfg():
    return SD15_lora_train(
        base_model='Lykon/DreamShaper',
        train_steps=1000,
        save_step=200,
        rank=8,
        dataset=dict(
            dataset1=cfg_data_SD_ARB(
                img_root='imgs/',
                batch_size=4,
                trigger_word='paimeng',
                resolution=512*512,
                num_bucket=4,
            )
        )
    )