import sys

sys.path.append('/')

import argparse
import os.path

import torch
from hcpdiff.utils.utils import str2bool
from hcpdiff.utils.net_utils import import_text_encoder_class, save_emb
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('pretrained_model_name_or_path', type=str)
    parser.add_argument('name', type=str)
    parser.add_argument('n_word', type=int, default=3)
    parser.add_argument('--init_text', type=str, default='*0.017', help='randn: *sigma (*0.5)')
    parser.add_argument('--root', type=str, default='embs/')
    parser.add_argument('--replace', type=str2bool, default=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    text_encoder_cls = import_text_encoder_class(args.pretrained_model_name_or_path, None)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None)

    embed_dim = text_encoder.text_model.embeddings.token_embedding.embedding_dim
    if args.init_text.startswith('*'):
        init_embs = torch.randn((args.n_word, embed_dim))
        if len(args.init_text) > 1:
            init_embs *= float(args.init_text[1:])
    else:
        emb_pt = text_encoder.text_model.embeddings.token_embedding
        prompt_ids = tokenizer(
            args.init_text, truncation=True, padding="max_length", return_tensors="pt",
            max_length=tokenizer.model_max_length).input_ids[0, 1:args.n_word + 1]
        init_embs = emb_pt(prompt_ids)
    print(init_embs.shape)
    save_emb(os.path.join(args.root, args.name + '.pt'), init_embs, args.replace)
    print(f'embedding {args.name} is create.')
