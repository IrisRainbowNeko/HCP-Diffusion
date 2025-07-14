"""
textencoder_ex.py
====================
    :Name:        extend text encoder
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from typing import Tuple, Optional

import torch
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from loguru import logger
from torch import nn
from transformers import CLIPTextModelWithProjection, T5EncoderModel
from transformers.models.clip.modeling_clip import CLIPAttention

class TEEXHook:
    def __init__(self, text_enc: nn.Module, tokenizer, N_repeats=1, clip_skip=0, clip_final_norm=True, use_attention_mask=False):
        self.text_enc = text_enc
        self.tokenizer = tokenizer

        self.N_repeats = N_repeats
        self.clip_skip = clip_skip
        self.clip_final_norm = clip_final_norm
        self.use_attention_mask = use_attention_mask

        text_enc.register_forward_hook(self.forward_hook)
        text_enc.register_forward_pre_hook(self.forward_hook_input)

    def find_final_norm(self, text_enc: nn.Module):
        for module in text_enc.modules():
            if 'final_layer_norm' in module._modules:
                logger.info(f'find final_layer_norm in {type(module)}')
                return module.final_layer_norm

        logger.info(f'final_layer_norm not found in {type(text_enc)}')
        return None

    @property
    def clip_final_norm(self):
        return self.final_layer_norm is not None

    @clip_final_norm.setter
    def clip_final_norm(self, value: bool):
        if value:
            self.final_layer_norm = self.find_final_norm(self.text_enc)
        else:
            self.final_layer_norm = None

    @property
    def N_repeats(self):
        return self.tokenizer.N_repeats

    @N_repeats.setter
    def N_repeats(self, value: int):
        self.tokenizer.N_repeats = value

    @property
    def device(self):
        return self.text_enc.device

    def forward_hook_input(self, host, feat_in):
        feat_re = rearrange(feat_in[0], 'b (r w) -> (b r) w', r=self.N_repeats)  # 使Attention mask的尺寸为N_word+2
        return (feat_re,) if len(feat_in) == 1 else (feat_re, *feat_in[1:])

    def forward_hook(self, host, feat_in: Tuple[torch.Tensor], feat_out):
        encoder_hidden_states = feat_out['hidden_states'][-self.clip_skip-1]
        if self.clip_final_norm and self.final_layer_norm is not None:
            encoder_hidden_states = self.final_layer_norm(encoder_hidden_states)
        if self.text_enc.training and self.clip_skip>0:
            encoder_hidden_states = encoder_hidden_states+0*feat_out['last_hidden_state'].mean()  # avoid unused parameters, make gradient checkpointing happy
        encoder_hidden_states = rearrange(encoder_hidden_states, '(b r) ... -> b r ...', r=self.N_repeats)  # [B, N_repeat, N_word+2, N_emb]
        pooled_output = feat_out.get('pooler_output', feat_out.get('text_embeds', None))
        # TODO: may have better fusion method
        if pooled_output is not None:
            pooled_output = rearrange(pooled_output, '(b r) ... -> b r ...', r=self.N_repeats).mean(dim=1)

        BOS, EOS = encoder_hidden_states[:, 0, :1, :], encoder_hidden_states[:, -1, -1:, :]
        encoder_hidden_states = torch.cat([BOS, encoder_hidden_states[:, :, 1:-1, :].flatten(1, 2), EOS], dim=1)  # [B, N_repeat*N_word+2, N_emb]

        return encoder_hidden_states, pooled_output

    def pool_hidden_states(self, encoder_hidden_states, input_ids):
        pooled_output = encoder_hidden_states[:, :, -1, :].mean(dim=1)  # [B, N_emb]
        return pooled_output

    @staticmethod
    def mult_attn(prompt_embeds, attn_mult):
        if attn_mult != None:
            for i, item in enumerate(attn_mult):
                if len(item)>0:
                    original_mean = prompt_embeds[i, ...].mean()
                    N_words = min(prompt_embeds.shape[1]-1, len(item))
                    prompt_embeds[i, 1:len(item)+1, :] *= item[:N_words].view(-1, 1).to(prompt_embeds.device)
                    new_mean = prompt_embeds[i, ...].mean()
                    prompt_embeds[i, ...] *= original_mean/new_mean
        return prompt_embeds

    def enable_xformers(self):
        try:
            from xformers.components import build_attention
            my_config = {
                "name":'scaled_dot_product',
                "dropout":0.0,
            }
            self.clip_attention = build_attention(my_config)

            for k, v in self.text_enc.named_modules():
                if isinstance(v, CLIPAttention):
                    self.apply_xformers_attention_clip(v)
        except:
            print('xformers not find')

    def apply_xformers_attention_clip(self, layer):
        re_qkv = Rearrange('b l (h hd) -> b h l hd', h=layer.num_heads, hd=layer.head_dim)

        def forward(hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    causal_attention_mask: Optional[torch.Tensor] = None,
                    output_attentions: Optional[bool] = False,
                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            """Input shape: Batch x Time x Channel"""

            # get query proj
            query_states = re_qkv(layer.q_proj(hidden_states))  # * layer.scale
            key_states = re_qkv(layer.k_proj(hidden_states))
            value_states = re_qkv(layer.v_proj(hidden_states))

            att_mask = None
            if attention_mask is not None:
                att_mask = repeat(attention_mask, 'b 1 x y -> (b h) x y', h=layer.num_heads)

            if causal_attention_mask is not None:
                causal_attention_mask = repeat(causal_attention_mask, 'b 1 x y -> (b h) x y', h=layer.num_heads)
                if att_mask is None:
                    att_mask = causal_attention_mask
                else:
                    att_mask += causal_attention_mask

            attn_output = self.clip_attention(query_states, key_states, value_states, att_mask=att_mask)
            attn_output = rearrange(attn_output, 'b h l hd -> b l (h hd)', h=layer.num_heads)

            attn_output = layer.out_proj(attn_output)

            return attn_output, None

        layer.forward = forward

    @classmethod
    def hook(cls, text_enc: nn.Module, tokenizer, N_repeats=3, clip_skip=0, clip_final_norm=True, use_attention_mask=False):
        return cls(text_enc, tokenizer, N_repeats=N_repeats, clip_skip=clip_skip, clip_final_norm=clip_final_norm,
                   use_attention_mask=use_attention_mask)

    @classmethod
    def hook_pipe(cls, pipe, N_repeats=3, clip_skip=0, clip_final_norm=True, use_attention_mask=False):
        return cls(pipe.text_encoder, pipe.tokenizer, N_repeats=N_repeats, clip_skip=clip_skip, clip_final_norm=clip_final_norm,
                   use_attention_mask=use_attention_mask)
