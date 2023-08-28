"""
textencoder_ex.py
====================
    :Name:        extend text encoder
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from typing import Tuple, Optional, List

import torch
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from torch import nn
from transformers.models.clip.modeling_clip import CLIPAttention

class TEEXHook:
    def __init__(self, text_enc: nn.Module, tokenizer, N_repeats=3, clip_skip=0, clip_final_norm=True, device='cuda'):
        self.text_enc = text_enc
        self.tokenizer = tokenizer

        self.N_repeats = N_repeats
        self.clip_skip = clip_skip
        self.clip_final_norm = clip_final_norm
        self.device = device
        self.attn_mult = None

        text_enc.register_forward_hook(self.forward_hook)
        text_enc.register_forward_pre_hook(self.forward_hook_input)

    def encode_prompt_to_emb(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length*self.N_repeats,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if hasattr(self.text_enc.config, "use_attention_mask") and self.text_enc.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None

        prompt_embeds, pooled_output = self.text_enc(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return prompt_embeds, pooled_output

    def forward_hook_input(self, host, feat_in):
        feat_re = rearrange(feat_in[0], 'b (r w) -> (b r) w', r=self.N_repeats)  # 使Attention mask的尺寸为N_word+2
        return (feat_re,) if len(feat_in) == 1 else (feat_re, *feat_in[1:])

    def forward_hook(self, host, feat_in: Tuple[torch.Tensor], feat_out):
        encoder_hidden_states = feat_out['hidden_states'][-self.clip_skip-1]
        if self.clip_final_norm:
            encoder_hidden_states = self.text_enc.text_model.final_layer_norm(encoder_hidden_states)
        if self.text_enc.training and self.clip_skip>0:
            encoder_hidden_states = encoder_hidden_states+0*feat_out['last_hidden_state']  # avoid unused parameters, make gradient checkpointing happy

        encoder_hidden_states = rearrange(encoder_hidden_states, '(b r) ... -> b r ...', r=self.N_repeats)  # [B, N_repeat, N_word+2, N_emb]
        pooled_output = feat_out.pooler_output
        BOS, EOS = encoder_hidden_states[:, 0, :1, :], encoder_hidden_states[:, -1, -1:, :]
        encoder_hidden_states = torch.cat([BOS, encoder_hidden_states[:, :, 1:-1, :].flatten(1, 2), EOS], dim=1)  # [B, N_repeat*N_word+2, N_emb]

        return encoder_hidden_states, pooled_output

    def pool_hidden_states(self, encoder_hidden_states, input_ids):
        pooled_output = encoder_hidden_states[:, :, -1, :].mean(dim=1) # [B, N_emb]
        return pooled_output

    @staticmethod
    def mult_attn(prompt_embeds, attn_mult):
        if attn_mult != None:
            for i, item in enumerate(attn_mult):
                if len(item)>0:
                    original_mean = prompt_embeds[i, ...].mean()
                    prompt_embeds[i, 1:len(item)+1, :] *= item.view(-1, 1).to(prompt_embeds.device)
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
    def hook(cls, text_enc: nn.Module, tokenizer, N_repeats=3, clip_skip=0, clip_final_norm=True, device='cuda'):
        return cls(text_enc, tokenizer, N_repeats, clip_skip, clip_final_norm, device)

    @classmethod
    def hook_pipe(cls, pipe, N_repeats=3, clip_skip=0, clip_final_norm=True):
        return cls(pipe.text_encoder, pipe.tokenizer, N_repeats=N_repeats, device='cuda', clip_skip=clip_skip, clip_final_norm=clip_final_norm)