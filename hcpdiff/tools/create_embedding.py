import sys

sys.path.append('/')

import argparse
import os.path

import torch
from hcpdiff.utils.utils import str2bool
from hcpdiff.utils.net_utils import auto_text_encoder_cls, save_emb, auto_tokenizer_cls, check_word_name
from hcpdiff.models.compose import ComposeTextEncoder

class PTCreator:
    def __init__(self, pretrained_model_name_or_path, root='embs/'):
        self.root = root

        tokenizer_cls = auto_tokenizer_cls(pretrained_model_name_or_path)
        self.tokenizer = tokenizer_cls.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        self.tokenizer.add_tokens('_pt_random_word')
        self.rand_holder_id = self.tokenizer('_pt_random_word', return_tensors="pt").input_ids.view(-1)[1].item()

        text_encoder_cls = auto_text_encoder_cls(pretrained_model_name_or_path)
        self.text_encoder = text_encoder_cls.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=None)

        emb_layer = self.text_encoder.get_input_embeddings()
        if isinstance(emb_layer, list):
            self.embed_dim = sum(layer.embedding_dim for layer in emb_layer)
        else:
            self.embed_dim = self.text_encoder.get_input_embeddings().embedding_dim

    def get_embs(self, prompt_ids):
        prompt_ids[prompt_ids == self.rand_holder_id] = 0
        emb_pt = self.text_encoder.get_input_embeddings()
        if isinstance(self.text_encoder, ComposeTextEncoder):
            prompt_ids_list = prompt_ids.chunk(len(self.text_encoder.model_names), dim=-1)
            emb_list = [layer(ids[1:-1]) for layer, ids in zip(emb_pt, prompt_ids_list)]
            return torch.cat(emb_list, dim=-1)
        else:
            return emb_pt(prompt_ids[1:-1])

    @staticmethod
    def find_random_holder(text, holder='_pt_random_word'):
        rand_list = []
        text_clean = ''

        sidx = text.find('*')
        eidx_last = 0
        while sidx != -1:
            eidx = text.find(']', sidx+2)
            rand_info = text[sidx+2:eidx].split(',')  # [std, len]
            if len(rand_info) == 1:
                rand_info = [rand_info[0], 1]
            rand_list.append([float(rand_info[0].strip()), int(rand_info[1].strip())])
            text_clean += text[eidx_last:sidx]+holder

            eidx_last = eidx+1
            sidx = text.find('*', eidx_last)
        text_clean += text[eidx_last:]

        return text_clean, rand_list

    def replace_random_holder(self, embs, prompt_ids, rand_list):
        if isinstance(self.text_encoder, ComposeTextEncoder):
            prompt_ids = prompt_ids.chunk(len(self.text_encoder.model_names), dim=-1)[0][1:-1]
        else:
            prompt_ids = prompt_ids[1:-1]

        idx = [-1]+torch.where(prompt_ids == self.rand_holder_id)[0].tolist()
        if len(idx) == 1:
            return embs

        emb_list = []
        for i in range(len(idx)-1):
            emb_list.append(embs[idx[i]+1:idx[i+1]])
            emb_list.append(torch.randn((rand_list[i][1], self.embed_dim))*rand_list[i][0])
        emb_list.append(embs[idx[-1]+1:])
        return torch.cat(emb_list, dim=0)

    def creat_word_pt(self, name, n_word, init_text, replace=False):
        check_word_name(self.tokenizer, name)
        text_clean, rand_list = self.find_random_holder(init_text)
        prompt_ids = self.tokenizer(
            text_clean, truncation=True, padding="max_length", return_tensors="pt",
            max_length=self.tokenizer.model_max_length).input_ids[0]
        init_embs = self.get_embs(prompt_ids.clone())
        init_embs = self.replace_random_holder(init_embs, prompt_ids, rand_list)
        init_embs = init_embs[:n_word, :]

        print(init_embs.shape)
        save_emb(os.path.join(self.root, name+'.pt'), init_embs, replace)
        print(f'embedding {name} is create.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('pretrained_model_name_or_path', type=str)
    parser.add_argument('name', type=str)
    parser.add_argument('n_word', type=int, default=3)
    parser.add_argument('--init_text', type=str, default='*[0.017,3]', help='randn: *[sigma,length]')
    parser.add_argument('--root', type=str, default='embs/')
    parser.add_argument('--replace', type=str2bool, default=False)
    args = parser.parse_args()

    pt_creator = PTCreator(args.pretrained_model_name_or_path, args.root)
    pt_creator.creat_word_pt(args.name, args.n_word, args.init_text, args.replace)
