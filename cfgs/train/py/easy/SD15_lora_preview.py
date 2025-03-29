from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SD15_lora_train, cfg_data_SD_ARB, SD15_t2i
from hcpdiff.evaluate import HCPPreviewer

prompt = ('paimeng, 1girl, halo, white_hair, solo, smile, blue_eyes, looking_at_viewer, open_mouth, long_sleeves, white_dress, dress, single_thighhigh,'
          ' :d, cape, hair_between_eyes, thighhighs, hair_ornament, blush, white_outline, outline, sky, scarf, cloud, white_thighhighs, arm_up,'
          ' notice_lines, paimon_(genshin_impact)')

@neko_cfg
def make_cfg():
    return dict(
        **SD15_lora_train(
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
        ),
        evaluator=HCPPreviewer(_partial_=True,
            interval=100,
            workflow=SD15_t2i(
                pretrained_model='${model.wrapper.models.ckpt_path}',
                prompt=prompt
            ),
        ),
    )