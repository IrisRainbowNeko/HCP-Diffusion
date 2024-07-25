import os
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from hcpdiff.utils import to_validate_file
from hcpdiff.utils.img_size_tool import types_support
from .hook import DiffusionHeatMapHooker
from ..base import ContainerAction, BasicAction, feedback_input

class CaptureCrossAttnAction(ContainerAction):
    def __init__(self, actions: List[BasicAction]):
        super().__init__(actions)

    @feedback_input
    def forward(self, memory, prompt, **states):
        bs = len(prompt)
        N_head = 8
        with DiffusionHeatMapHooker(memory.unet, memory.tokenizer, vae_scale_factor=memory.vae.vae_scale_factor) as tc:
            states = self.inner_forward(memory, **states)
            heat_maps = [tc.compute_global_heat_map(prompt=prompt[i], head_idxs=range(N_head*i, N_head*(i+1))) for i in range(bs)]

        return {**states, 'cross_attn_heat_maps':heat_maps}

class SaveWordAttnAction(BasicAction):

    def __init__(self, save_root: str, N_col: int = 4, image_type: str = 'png', quality: int = 95):
        self.save_root = save_root
        self.image_type = image_type
        self.quality = quality
        self.N_col = N_col

        os.makedirs(save_root, exist_ok=True)

    def draw_attn(self, tokenizer, prompt, image, global_heat_map):
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
            heat_map.plot_overlay(image, ax=ax[i//d_len, i%d_len])
        # plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(buf)

    @feedback_input
    def forward(self, memory, images, prompt, seeds, cross_attn_heat_maps, **states):
        num_img_exist = max([0]+[int(x.split('-', 1)[0]) for x in os.listdir(self.save_root) if x.rsplit('.', 1)[-1] in types_support])

        for bid, (p, img) in enumerate(zip(prompt, images)):
            img_path = os.path.join(self.save_root, f"{num_img_exist}-{seeds[bid]}-cross_attn-{to_validate_file(prompt[0])}.{self.image_type}")
            img = self.draw_attn(memory.tokenizer, p, img, cross_attn_heat_maps[bid])
            img.save(img_path, quality=self.quality)
            num_img_exist += 1
