"""
text_emb_ex.py
====================
    :Name:        hook for embedding
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""
from typing import Tuple

import torch
from torch import nn
import os
from loguru import logger
from einops import rearrange, repeat

from ..utils.net_utils import load_emb
from .plugin import SinglePluginBlock

class EmbeddingPTHook(SinglePluginBlock):
    def __init__(self, token_embedding:nn.Embedding, N_word=75, N_repeats=3):
        super().__init__('emb_ex', token_embedding)
        self.handle_pre = token_embedding.register_forward_pre_hook(self.pre_hook)

        self.N_word=N_word
        self.N_repeats=N_repeats
        self.num_embeddings=token_embedding.num_embeddings
        self.embedding_dim=token_embedding.embedding_dim
        self.emb={}
        self.emb_train=nn.ParameterList()

    def add_emb(self, emb:nn.Parameter, token_id:int):
        self.emb[token_id]=emb

    def pre_hook(self, host, input_ids: Tuple[torch.Tensor]):
        self.input_ids = rearrange(input_ids[0], '(b r) w -> b (r w)', r=self.N_repeats)  # 兼容Attention mask
        return self.input_ids.clip(0, self.num_embeddings-1)

    def forward(self, fea_in:Tuple[torch.Tensor], inputs_embeds:torch.Tensor):
        '''
        :param input_ids: [B, N_ids]
        :param inputs_embeds: [B, N_repeat*(N_word+2), N_emb]
        :return: [B, N_repeat, N_word+2, N_emb]
        '''
        rep_idxs_B = self.input_ids >= self.num_embeddings
        BOS = repeat(inputs_embeds[0,0,:], 'e -> r 1 e', r=self.N_repeats)
        EOS = repeat(inputs_embeds[0,-1,:], 'e -> r 1 e', r=self.N_repeats)

        replaced_embeds = []
        for item, rep_idxs, ids_raw in zip(inputs_embeds, rep_idxs_B, self.input_ids):
            # insert pt to embeddings
            rep_idxs=torch.where(rep_idxs)[0]
            item_new=[]
            rep_idx_last=0
            for rep_idx in rep_idxs:
                rep_idx=rep_idx.item()
                item_new.append(item[rep_idx_last:rep_idx, :])
                item_new.append(self.emb[ids_raw[rep_idx].item()].to(dtype=item.dtype))
                rep_idx_last=rep_idx+1
            item_new.append(item[rep_idx_last:, :])

            # split to N_repeat sentence
            replaced_item = torch.cat(item_new, dim=0)[1:self.N_word*self.N_repeats+1, :]
            replaced_item = rearrange(replaced_item, '(r w) e -> r w e', r=self.N_repeats, w=self.N_word)
            replaced_item = torch.cat([BOS, replaced_item, EOS], dim=1) # [N_repeat, N_word+2, N_emb]

            replaced_embeds.append(replaced_item)
        return torch.cat(replaced_embeds, dim=0) # [B*N_repeat, N_word+2, N_emb]

    def remove(self):
        super(EmbeddingPTHook, self).remove()
        self.handle_pre.remove()

    @classmethod
    def hook(cls, ex_words_emb, tokenizer, text_encoder, log=False, **kwargs):
        word_list = list(ex_words_emb.keys())
        tokenizer.add_tokens(word_list)
        token_ids = tokenizer(' '.join(word_list)).input_ids[1:-1]

        embedding_hook = cls(text_encoder.text_model.embeddings.token_embedding, **kwargs)
        #text_encoder.text_model.embeddings.token_embedding = embedding_hook
        for tid, word in zip(token_ids, word_list):
            embedding_hook.add_emb(ex_words_emb[word], tid)
            if log:
                logger.info(f'hook: {word}, len: {ex_words_emb[word].shape[0]}, id: {tid}')
        return embedding_hook

    @classmethod
    def hook_from_dir(cls, emb_dir, tokenizer, text_encoder, log=True, device='cuda:0', **kwargs):
        ex_words_emb = {file[:-3]: nn.Parameter(load_emb(os.path.join(emb_dir, file)).to(device), requires_grad=False)
                        for file in os.listdir(emb_dir) if file.endswith('.pt')}
        return cls.hook(ex_words_emb, tokenizer, text_encoder, log, **kwargs), ex_words_emb

