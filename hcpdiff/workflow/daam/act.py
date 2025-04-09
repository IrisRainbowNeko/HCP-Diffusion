import os
from io import BytesIO

import numpy as np
from PIL import Image
from hcpdiff.utils import to_validate_file
from rainbowneko.utils import types_support
from matplotlib import pyplot as plt
from rainbowneko.infer import BasicAction, Actions

from .hook import DiffusionHeatMapHooker

class CaptureCrossAttnAction(Actions):
    def forward(self, prompt, denoiser, tokenizer, vae, **states):
        bs = len(prompt)
        N_head = 8
        with DiffusionHeatMapHooker(denoiser, tokenizer, vae_scale_factor=vae.vae_scale_factor) as tc:
            states = super().forward(**states)
            heat_maps = [tc.compute_global_heat_map(prompt=prompt[i], head_idxs=range(N_head*i, N_head*(i+1))) for i in range(bs)]

        return {**states, 'cross_attn_heat_maps':heat_maps}

class SaveWordAttnAction(BasicAction):

    def __init__(self, save_root: str, N_col: int = 4, image_type: str = 'png', quality: int = 95, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.save_root = save_root
        self.image_type = image_type
        self.quality = quality
        self.N_col = N_col

        os.makedirs(save_root, exist_ok=True)

    def draw_attn(self, tokenizer, prompt, image, global_heat_map):
        prompt=tokenizer.bos_token+prompt+tokenizer.eos_token
        tokens = [token.replace("</w>", "") for token in tokenizer.tokenize(prompt)]

        d_len = self.N_col
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams.update({'font.size':12})
        h = int(np.ceil(len(tokens)/d_len))
        fig, ax = plt.subplots(h, d_len, figsize=(2*d_len, 2*h))
        for ax_ in ax.flatten():
            ax_.set_xticks([])
            ax_.set_yticks([])
        for i, token in enumerate(tokens):
            heat_map = global_heat_map.compute_word_heat_map(token, word_idx=i)
            if h==1:
                heat_map.plot_overlay(image, ax=ax[i%d_len])
            else:
                heat_map.plot_overlay(image, ax=ax[i//d_len, i%d_len])
        # plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(buf)

    def forward(self, tokenizer, images, prompt, seeds, cross_attn_heat_maps, **states):
        num_img_exist = max([0]+[int(x.split('-', 1)[0]) for x in os.listdir(self.save_root) if x.rsplit('.', 1)[-1] in types_support])

        for bid, (p, img) in enumerate(zip(prompt, images)):
            img_path = os.path.join(self.save_root, f"{num_img_exist}-{seeds[bid]}-cross_attn-{to_validate_file(prompt[0])}.{self.image_type}")
            img = self.draw_attn(tokenizer, p, img, cross_attn_heat_maps[bid])
            img.save(img_path, quality=self.quality)
            num_img_exist += 1
