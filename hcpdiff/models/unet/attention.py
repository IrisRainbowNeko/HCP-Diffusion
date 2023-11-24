from torch import nn
import torch
from typing import Optional
from einops import rearrange

try:
    import xformers
    import xformers.ops
except:
    xformers = None

class Attention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int=None, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, bias: bool = False,
                 out_bias: bool = True, scale_qk: bool = True):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head*heads
        self.q_dim = q_dim
        self.kv_dim = kv_dim or q_dim
        self.dropout = dropout
        self.scale = dim_head**-0.5 if scale_qk else 1.0

        self.to_q = nn.Linear(self.q_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.kv_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.kv_dim, self.inner_dim, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.q_dim, bias=out_bias),
            nn.Dropout(dropout)
        )

    def prepare_attention_mask(self, attention_mask: torch.Tensor, batch_size: int, out_dim: int = 3) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def scale_dot_product(self, query, key, value, attention_mask):
        # Q * K^T
        attention_scores = torch.baddbmm(attention_mask, query, key.transpose(1, 2), alpha=self.scale)
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        # attn * V
        hidden_states = torch.bmm(attention_probs, value)
        return hidden_states

    def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None):
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = rearrange(self.to_q(hidden_states), 'b l (h d) -> (b h) l d', h=self.heads)
        key = rearrange(self.to_k(encoder_hidden_states), 'b l (h d) -> (b h) l d', h=self.heads)
        value = rearrange(self.to_v(encoder_hidden_states), 'b l (h d) -> (b h) l d', h=self.heads)

        hidden_states = self.scale_dot_product(query, key, value, attention_mask)

        hidden_states = rearrange(hidden_states, '(b h) l d -> b l (h d)', h=self.heads)
        hidden_states = self.to_out(hidden_states)
        return hidden_states

class AttentionXFormers(Attention):
    def __init__(self, *args, attention_op=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_op = attention_op

    def scale_dot_product(self, query, key, value, attention_mask):
        return xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=self.scale)