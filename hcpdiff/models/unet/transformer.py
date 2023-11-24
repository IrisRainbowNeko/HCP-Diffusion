from typing import Optional, Dict, Any

import torch
from torch import nn
from .layers import GEGLU, GELU, ApproximateGELU
from functools import partial
from .attention import Attention
from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, activation_builder: partial = partial(GEGLU), dropout: float = 0.0, ):
        super().__init__()
        inner_dim = int(dim*mult)
        dim_out = dim_out or dim

        self.net = nn.Sequential(
            activation_builder(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)

class Transformer2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dim_head, dropout=0.0, cross_attention_dim: Optional[int] = None, qkv_bias:bool=False,
                 activation_builder: partial = partial(GEGLU), norm_builder: Optional[partial] = None, in_norm_builder: Optional[partial] = None):
        super().__init__()
        inner_dim = heads*dim_head

        if in_norm_builder is None:
            in_norm_builder = partial(nn.GroupNorm, num_groups=32, eps=1e-6, affine=True)

        self.norm_in = in_norm_builder(num_channels=in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        if norm_builder is None:
            norm_builder = partial(nn.LayerNorm, elementwise_affine=True, eps=1e-5)

        self.norm1 = norm_builder(inner_dim)
        self.attn1 = Attention(q_dim=inner_dim, heads=heads, dim_head=dim_head, dropout=dropout, bias=qkv_bias)
        self.norm2 = norm_builder(inner_dim)
        self.attn2 = Attention(q_dim=inner_dim, kv_dim=cross_attention_dim, heads=heads, dim_head=dim_head, dropout=dropout, bias=qkv_bias)
        self.norm3 = norm_builder(inner_dim)
        self.ff = FeedForward(inner_dim, dropout=dropout, activation_builder=activation_builder)

        self.proj_out = nn.Conv2d(inner_dim, out_channels, kernel_size=1, stride=1, padding=0)

    def transformer_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # self attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output

        # cross attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, attention_mask=encoder_attention_mask, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # feed forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output

        return hidden_states

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')

        hidden_states = self.transformer_forward(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)

        hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h=residual.shape[2]).contiguous()
        hidden_states = self.proj_out(hidden_states)
        return hidden_states + residual
