from daam import AggregateHooker, RawHeatMapCollection, UNetCrossAttentionLocator, GlobalHeatMap
from daam.trace import UNetCrossAttentionHooker
from typing import List
from diffusers import UNet2DConditionModel
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def auto_autocast(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs['enabled'] = False

    return torch.cuda.amp.autocast(*args, **kwargs)

class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(
            self,
            unet: UNet2DConditionModel,
            tokenizer,
            vae_scale_factor: int,
            low_memory: bool = False,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: str = None
    ):
        self.all_heat_maps = RawHeatMapCollection()
        h = (unet.config.sample_size * vae_scale_factor)
        self.latent_hw = 4096 if h == 512 or h == 1024 else 9216  # 64x64 or 96x96 depending on if it's 2.0-v or 2.0
        locate_middle = load_heads or save_heads
        self.locator = UNetCrossAttentionLocator(restrict={0} if low_memory else None, locate_middle_block=locate_middle)
        self.last_prompt: str = ''
        self.last_image: Image.Image = None
        self.time_idx = 0
        self._gen_idx = 0

        self.tokenizer = tokenizer

        modules = [
            UNetCrossAttentionHooker(
                x,
                self,
                layer_idx=idx,
                latent_hw=self.latent_hw,
                load_heads=load_heads,
                save_heads=save_heads,
                data_dir=data_dir
            ) for idx, x in enumerate(self.locator.locate(unet))
        ]

        super().__init__(modules)

    def time_callback(self, *args, **kwargs):
        self.time_idx += 1

    @property
    def layer_names(self):
        return self.locator.layer_names

    def compute_global_heat_map(self, prompt=None, factors=None, head_idxs: List[int]=None, layer_idx=None, normalize=False):
        # type: (str, List[float], int, int, bool) -> GlobalHeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        heat_maps = self.all_heat_maps

        if prompt is None:
            prompt = self.last_prompt

        if factors is None:
            factors = {0, 1, 2, 4, 8, 16, 32, 64}
        else:
            factors = set(factors)

        all_merges = []
        x = int(np.sqrt(self.latent_hw))

        with auto_autocast(dtype=torch.float32):
            for (factor, layer, head), heat_map in heat_maps:
                if (head_idxs is None or head in head_idxs) and (layer_idx is None or layer_idx == layer):
                    heat_map = heat_map.unsqueeze(1)/25 # [L,1,H,W]
                    # The clamping fixes undershoot.
                    all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))

            try:
                maps = torch.stack(all_merges, dim=0) # [B*head, L, 1, H, W]
            except RuntimeError:
                if head_idxs is not None or layer_idx is not None:
                    raise RuntimeError('No heat maps found for the given parameters.')
                else:
                    raise RuntimeError('No heat maps found. Did you forget to call `with trace(...)` during generation?')

            maps = maps.mean(0)[:, 0] # [L,H,W]
            #maps = maps[:len(self.tokenizer.tokenize(prompt)) + 2]  # 1 for SOS and 1 for padding

            if normalize:
                maps = maps / (maps[1:-1].sum(0, keepdim=True) + 1e-6)  # drop out [SOS] and [PAD] for proper probabilities

        return GlobalHeatMap(self.tokenizer, prompt, maps)