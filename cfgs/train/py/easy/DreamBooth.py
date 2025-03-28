from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SD15_finetuning, cfg_data_SD_ARB, cfg_data_SD_resize_crop

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
                word_names={
                    'pt1':'[V]',
                    'class':'dog'
                }
            ),
            dataset_class=cfg_data_SD_resize_crop(
                img_root='imgs_db_class/',
                batch_size=1,
                target_size=[512, 512],
                word_names={'class':'dog'}
            )
        )
    )