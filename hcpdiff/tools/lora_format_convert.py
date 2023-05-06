from typing import Dict, List, Iterable, Tuple, Union

import re
import torch
from torch import nn

from hcpdiff.utils.ckpt_manager import CkptManagerPKL, CkptManagerSafe
import argparse

def change_unet(sd_fold):
    com_name = ['down_blocks','up_blocks', 'mid_block', 'attentions','transformer_blocks','attn1','attn2',
                'to_q','to_k','to_v','to_out','lora_down','lora_up','weight','ff','net','proj_in','proj_out']
    
    keys = sd_fold.keys()
    new_sd_fold = dict()
    for key in keys:
        if 'text' in key:
            continue
        s = key.split('.')
        a = s[0]
        b = '.'.join(s[1:])
        sa = a.split('_')
        csa = []
        i = 2
        while i < len(sa):
            if sa[i] in com_name:
                csa.append(sa[i])
                i+=1
            elif i+1 < len(sa) and sa[i]+'_'+sa[i+1] in com_name:
                csa.append(sa[i]+'_'+sa[i+1])
                i+=2
            elif sa[i].isnumeric():
                csa.append(sa[i])
                i+=1
            elif 'ff' in sa and sa[i] == 'proj':
                csa.append(sa[i])
                i+=1          
            else:
                print('wwww',sa[i])
        new_key = '.'.join(csa)
        if 'alpha' in b:
            new_key = new_key + '.lora_block.scale'
        else:
            new_key = new_key + '.lora_block.layer.' + b
        new_sd_fold[new_key] = sd_fold[key]
    new_sd_fold = {'lora':new_sd_fold}
    return new_sd_fold

def change_text_encoder(sd_fold):
    com_name = ['text_model','encoder', 'layers', 'self_attn','q_proj','v_proj','k_proj','out_proj',
                'lora_down','lora_up','weight','mlp','fc1','fc2']
    keys = sd_fold.keys()
    new_sd_fold = dict()
    for key in keys:
        if 'block' in key:
            continue
        s = key.split('.')
        a = s[0]
        b = '.'.join(s[1:])
        sa = a.split('_')
        csa = []
        i = 2
        while i < len(sa):
            if sa[i] in com_name:
                csa.append(sa[i])
                i+=1
            elif i+1 < len(sa) and sa[i]+'_'+sa[i+1] in com_name:
                csa.append(sa[i]+'_'+sa[i+1])
                i+=2
            elif sa[i].isnumeric():
                csa.append(sa[i])
                i+=1        
            else:
                print('wwww',sa[i])
        new_key = '.'.join(csa)
        if 'alpha' in b:
            new_key = new_key + '.lora_block.scale'
        else:
            new_key = new_key + '.lora_block.layer.' + b
        new_sd_fold[new_key] = sd_fold[key]
    new_sd_fold = {'lora':new_sd_fold}
    return new_sd_fold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_path", default=None, type=str, required=True, help="Path to the lora.safetensor to convert.")
    parser.add_argument("--dump_unet_path", default='conveted_unet.safetensors', type=str, required=False, help="Path to save the converted unet")
    parser.add_argument("--dump_text_encoder_path", default='conveted_text_encoder.safetensors', type=str, required=False, help="Path to save the converted text encoder")

    args = parser.parse_args()

    # load lora model
    ckpt_manager_safe = CkptManagerSafe()
    sd_fold = ckpt_manager_safe.load_ckpt('/home/amax/hzj/stable-diffusion-webui/models/Lora/keqingGenshinImpact3in1_v10.safetensors')
    sd_fold = ckpt_manager_safe.load_ckpt(args.lora_model_path)
    # convert the weight name
    unet_sd_fold = change_unet(sd_fold)
    text_encoder_sd_fold = change_text_encoder(sd_fold)
    # wegiht save
    # ckpt_manager_safe.set_save_dir('./')
    ckpt_manager_safe._save_ckpt(unet_sd_fold, 'unet', 0, save_path=args.dump_unet_path)
    ckpt_manager_safe._save_ckpt(text_encoder_sd_fold, 'text_encoder', 0, save_path=args.dump_text_encoder_path)