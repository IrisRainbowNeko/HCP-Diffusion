from .text2img import *

@neko_cfg
def make_cfg():
    return dict(
        pretrained_model='Lykon/DreamShaper',
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=None,
        bs=4,

        workflow=Actions(actions=[
            build_model(pretrained_model='${pretrained_model}'),
            optimize_model(),
            text(prompt='${prompt}', negative_prompt='${negative_prompt}', bs='${bs}'),
            config_diffusion(seed='${seed}'),
            diffusion(),
            decode()
        ])
    )
