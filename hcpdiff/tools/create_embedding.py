import sys

sys.path.append('/')

import argparse
import os.path

import torch
from hcpdiff.utils.utils import str2bool
from hcpdiff.utils.net_utils import auto_text_encoder, save_emb, auto_tokenizer
from hcpdiff.models.compose import ComposeTextEncoder

class PTCreator:
    def __init__(self, pretrained_model_name_or_path, root='embs/'):
        self.root=root

        tokenizer_cls = auto_tokenizer(pretrained_model_name_or_path)
        self.tokenizer = tokenizer_cls.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

        text_encoder_cls = auto_text_encoder(pretrained_model_name_or_path)
        self.text_encoder = text_encoder_cls.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=None)

        emb_layer = self.text_encoder.get_input_embeddings()
        if isinstance(emb_layer, list):
            self.embed_dim = sum(layer.embedding_dim for layer in emb_layer)
        else:
            self.embed_dim = self.text_encoder.get_input_embeddings().embedding_dim

    def get_embs(self, prompt_ids, n_word):
        emb_pt = self.text_encoder.get_input_embeddings()
        if isinstance(self.text_encoder, ComposeTextEncoder):
            prompt_ids_list = prompt_ids.chunk(len(self.text_encoder.model_names),dim=-1)
            emb_list = [layer(ids[0, 1:n_word+1]) for layer, ids in zip(emb_pt, prompt_ids_list)]
            return torch.cat(emb_list, dim=-1)
        else:
            return emb_pt(prompt_ids[0, 1:n_word+1])

    def creat_word_pt(self, name, n_word, init_text, replace=False):
        if init_text.startswith('*'):
            init_embs = torch.randn((n_word, self.embed_dim))
            if len(init_text)>1:
                init_embs *= float(init_text[1:])
        else:
            prompt_ids = self.tokenizer(
                init_text, truncation=True, padding="max_length", return_tensors="pt",
                max_length=self.tokenizer.model_max_length).input_ids#[0, 1:n_word+1]
            init_embs = self.get_embs(prompt_ids, n_word)
        print(init_embs.shape)
        save_emb(os.path.join(self.root, name+'.pt'), init_embs, replace)
        print(f'embedding {name} is create.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('pretrained_model_name_or_path', type=str)
    parser.add_argument('name', type=str)
    parser.add_argument('n_word', type=int, default=3)
    parser.add_argument('--init_text', type=str, default='*0.017', help='randn: *sigma (*0.5)')
    parser.add_argument('--root', type=str, default='embs/')
    parser.add_argument('--replace', type=str2bool, default=False)
    args = parser.parse_args()

    pt_creator = PTCreator(args.pretrained_model_name_or_path, args.root)
    pt_creator.creat_word_pt(args.name, args.n_word, args.init_text, args.replace)
