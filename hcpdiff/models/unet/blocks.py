from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .layers import Downsample2D, Upsample2D
from .resnet import ResnetBlock
from .transformer import Transformer2DBlock

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, dropout: float = 0.0, num_layers: int = 1,
                 add_downsample: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gradient_checkpointing = False

        resnets = []
        for i in range(num_layers):
            resnets.append(ResnetBlock(in_channels, out_channels, temb_channels=temb_channels, dropout=dropout))
            in_channels = out_channels
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = Downsample2D(out_channels)
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()
        if self.training and self.gradient_checkpointing:
            for resnet in self.resnets:
                hidden_states = torch.utils.checkpoint.checkpoint(resnet, hidden_states, temb)
                output_states = output_states+(hidden_states,)
        else:
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states, temb)
                output_states = output_states+(hidden_states,)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers(hidden_states)
            output_states = output_states+(hidden_states,)
        return hidden_states, output_states

class CrossAttnDownBlock(DownBlock):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, dropout: float = 0.0, num_layers: int = 1,
                 add_downsample: bool = True, heads: int = 1, cross_attention_dim: int = 1280):
        super().__init__(in_channels, out_channels, temb_channels, dropout, num_layers, add_downsample)
        attentions = []
        for i in range(num_layers):
            attentions.append(Transformer2DBlock(out_channels, out_channels, heads=heads, dim_head=out_channels//heads,
                                                 cross_attention_dim=cross_attention_dim))
        self.attentions = nn.ModuleList(attentions)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()
        if self.training and self.gradient_checkpointing:
            for resnet, attn in zip(self.resnets, self.attentions):
                hidden_states = checkpoint(resnet, hidden_states, temb)
                hidden_states = checkpoint(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                                           attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask)
                output_states = output_states+(hidden_states,)
        else:
            for resnet, attn in zip(self.resnets, self.attentions):
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
                                     encoder_attention_mask=encoder_attention_mask)
                output_states = output_states+(hidden_states,)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers(hidden_states)
            output_states = output_states+(hidden_states,)
        return hidden_states, output_states

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, prev_out_ch:int, temb_channels: int, dropout: float = 0.0, num_layers: int = 1,
                 add_upsample: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gradient_checkpointing = False

        resnets = []
        for i in range(num_layers):
            res_skip_channels = prev_out_ch if (i == num_layers-1) else out_channels
            resnet_in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock(resnet_in_channels + res_skip_channels, out_channels, temb_channels=temb_channels, dropout=dropout))
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = Upsample2D(out_channels, out_channels)
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        upsample_size=None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        if self.training and self.gradient_checkpointing:
            for resnet in self.resnets:
                # unet skip connection
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                hidden_states = torch.utils.checkpoint.checkpoint(resnet, hidden_states, temb)
        else:
            for resnet in self.resnets:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            hidden_states = self.upsamplers(hidden_states, upsample_size)
        return hidden_states

class CrossAttnUpBlock(UpBlock):
    def __init__(self, in_channels: int, out_channels: int, prev_out_ch:int, temb_channels: int, dropout: float = 0.0, num_layers: int = 1,
                 add_upsample: bool = True, heads: int = 1, cross_attention_dim: int = 1280):
        super().__init__(in_channels, out_channels, prev_out_ch, temb_channels, dropout, num_layers, add_upsample)
        attentions = []
        for i in range(num_layers):
            attentions.append(Transformer2DBlock(out_channels, out_channels, heads=heads, dim_head=out_channels//heads,
                                                 cross_attention_dim=cross_attention_dim))
        self.attentions = nn.ModuleList(attentions)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        upsample_size = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        if self.training and self.gradient_checkpointing:
            for resnet, attn in zip(self.resnets, self.attentions):
                # unet skip connection
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                hidden_states = checkpoint(resnet, hidden_states, temb)
                hidden_states = checkpoint(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                                           attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask)
        else:
            for resnet, attn in zip(self.resnets, self.attentions):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
                                     encoder_attention_mask=encoder_attention_mask)

        if self.upsamplers is not None:
            hidden_states = self.upsamplers(hidden_states, upsample_size)
        return hidden_states

class CrossAttnMidBlock(CrossAttnDownBlock):
    def __init__(self, in_channels: int, temb_channels: int, dropout: float = 0.0, num_layers: int = 1,
                heads: int = 1, cross_attention_dim: int = 1280):
        super().__init__(in_channels, in_channels, temb_channels, dropout, num_layers, False, heads, cross_attention_dim)
        self.resnet0 = ResnetBlock(in_channels, in_channels, temb_channels=temb_channels, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        if self.training and self.gradient_checkpointing:
            hidden_states = checkpoint(self.resnet0, hidden_states, temb)
            for resnet, attn in zip(self.resnets, self.attentions):
                hidden_states = checkpoint(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                                           attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask)
                hidden_states = checkpoint(resnet, hidden_states, temb)
        else:
            hidden_states = self.resnet0(hidden_states, temb)
            for resnet, attn in zip(self.resnets, self.attentions):
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
                                     encoder_attention_mask=encoder_attention_mask)
                hidden_states = resnet(hidden_states, temb)

        return hidden_states