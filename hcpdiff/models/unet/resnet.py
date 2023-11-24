from functools import partial
from typing import Optional

from torch import nn

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_builder: Optional[partial] = None, temb_channels: int = 512, dropout: float = 0.0,
                 conv_shortcut_bias=False):
        super().__init__()
        if norm_builder is None:
            norm_builder = partial(nn.GroupNorm, num_groups=32, eps=1e-5, affine=True)

        self.block1 = nn.Sequential(
            norm_builder(num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_channels, out_channels),
        )
        self.block2 = nn.Sequential(
            norm_builder(num_channels=out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias)
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, x, temb):
        shortcut = self.conv_shortcut(x)
        x = self.block1(x)
        x = x+self.time_emb_proj(temb)
        x = self.block2(x)
        return x+shortcut
