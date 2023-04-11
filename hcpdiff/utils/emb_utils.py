import torch
import os

def load_emb(path):
    emb=torch.load(path)['string_to_param']['*']
    emb.requires_grad_(False)
    return emb

def save_emb(path, emb:torch.Tensor, replace=False):
    name = os.path.basename(path)
    if os.path.exists(path) and not replace:
        raise FileExistsError(f'embedding "{name}" already exist.')
    name=name[:name.rfind('.')]
    torch.save({'string_to_param':{'*':emb}, 'name':name}, path)