import sys
sys.path.append('./')

import torch
from torch import nn
import argparse

from diffusers import UNet2DConditionModel, StableDiffusionPipeline

from models.lora import collapse_lora_weight
from utils.cfg_net_tools import get_layers_with_block, get_match_layers
from utils.utils import str2bool, load_config, import_model_class_from_model_name_or_path

def get_blocks_from_lora_state(state_dict):
    return list({name[:name.rfind('lora_block.')-1] for name in state_dict.keys() if name.rfind('lora_block.')!=-1})

def merge_to_base_model(base_model:nn.Module, cfg_part):
    keys_merged = set()
    base_alpha = cfg_part.base_model_alpha
    named_modules = {k: v for k, v in base_model.named_modules()}
    all_layers = list(named_modules.keys())
    base_state = {k:v*base_alpha for k,v in base_model.state_dict().items()}

    def add_to_base(state, alpha):
        for k,v in state.items():
            base_state[k]+=alpha*v
            keys_merged.add(k)

    if cfg_part.lora is not None:
        for item in cfg_part.lora:
            lora_state=torch.load(item.path, map_location='cpu')['lora']
            lora_block_state = {}
            # get all layers in the lora_state
            for name, p in lora_state.items():
                lbidx = name.rfind('lora_block.')
                if lbidx != -1:
                    prefix = name[:lbidx-1]
                    if prefix not in lora_block_state:
                        lora_block_state[prefix]={}
                    lora_block_state[prefix][name[lbidx+len('lora_block.'):]]=p
            # get selected layers
            if item.layers != 'all':
                match_blocks = get_match_layers(item.layers, named_modules)
                match_layers = get_layers_with_block(named_modules, match_blocks, [nn.Linear, nn.Conv2d])
                lora_block_state = {k:v for k,v in lora_block_state.items() if k in match_layers}
            # collapse lora weight and add to dict
            lora_state_add={}
            for k, v in lora_block_state.items():
                lora_state_add[f'{k}.weight']=v['scale']*collapse_lora_weight(v['lora_up.weight'], v['lora_down.weight'],
                                                    'linear' if isinstance(named_modules[k], nn.Linear) else 'conv')
                if 'lora_up.bias' in v:
                    if named_modules[k].bias is None:
                        named_modules[k].bias = nn.Parameter(torch.zeros_like(v['lora_up.bias']))
                    lora_state_add[f'{k}.bias'] = v['lora_up.bias']
            add_to_base(lora_state_add, item.alpha)

    if cfg_part.part is not None:
        for item in cfg_part.part:
            part_state = torch.load(item.path, map_location='cpu')['base']
            if item.layers == 'all':
                add_to_base(part_state, item.alpha)
            else:
                match_blocks = get_match_layers(item.layers, named_modules)
                state_add = {k:v for blk in match_blocks for k,v in part_state.items() if k.startswith(blk)}
                add_to_base(state_add, item.alpha)

    # remove unused state
    base_state={k:base_state[k] for k in keys_merged}
    base_model.load_state_dict(base_state, strict=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfgs/infer/v1.yaml')
    parser.add_argument('--base_model', type=str, default='cfgs/infer/v1.yaml')
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet", revision=args.revision)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.base_model, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(args.base_model, subfolder="text_encoder", revision=args.revision)

    cfgs = load_config(args.cfg)
    for cfg_group in cfgs.values():
        if hasattr(cfg_group, 'type'):
            if cfg_group.type=='unet':
                merge_to_base_model(unet, cfg_group)
            elif cfg_group.type=='TE':
                merge_to_base_model(text_encoder, cfg_group)

    comp = StableDiffusionPipeline.from_pretrained(args.base_model).components
    comp['unet'] = unet
    comp['text_encoder'] = text_encoder
    pipe = StableDiffusionPipeline(**comp)
    pipe.save_pretrained(args.out_dir)