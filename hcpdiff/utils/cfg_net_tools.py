"""
cfg_net_tools.py
====================
    :Name:        creat model and plugin from config
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""


from typing import Dict, List, Iterable, Tuple, Union

import re
import torch
from torch import nn

from .utils import dict_get
from hcpdiff.models.lora_base import LoraBlock, LoraGroup
from hcpdiff.models import lora_layers
from hcpdiff.models.plugin import SinglePluginBlock, MultiPluginBlock, PluginBlock, PluginGroup
from .ckpt_manager import CkptManagerPKL, CkptManagerSafe

def get_class_match_layer(class_name, block:nn.Module):
    if type(block).__name__==class_name:
        return ['']
    else:
        return ['.'+name for name, layer in block.named_modules() if type(layer).__name__==class_name]

def get_match_layers(layers, all_layers):
    res=[]
    for name in layers:
        if isinstance(name, str):
            if name.startswith('re:'):
                pattern = re.compile(name[3:])
                res.extend(filter(lambda x: pattern.match(x) != None, all_layers.keys()))
            else:
                res.append(name)
        else:
            pattern = re.compile(name[0][3:])
            match_layers = filter(lambda x: pattern.match(x) != None, all_layers.keys())
            for layer in match_layers:
                res.extend([layer+x for x in get_class_match_layer(name[1], all_layers[layer])])

    return sorted(set(res), key=res.index) # Remove duplicates and keep the original order

def get_layers_with_block(named_modules:Dict[str, nn.Module], block_names:Iterable[str], cond:List=None):
    layers=[]
    for blk in block_names:
        if type(named_modules[blk]) in cond:
            layers.append(blk)
        for name, layer in named_modules[blk].named_modules():
            if type(named_modules[blk]) in cond:
                layers.append(f'{blk}.{name}')
    return layers

def make_hcpdiff(model, cfg_model, cfg_lora, default_lr=1e-5) -> Tuple[List[Dict], Union[LoraGroup, Tuple[LoraGroup, LoraGroup]]]:
    named_modules = {k:v for k,v in model.named_modules()}

    train_params=[]
    all_lora_blocks={}
    all_lora_blocks_neg={}

    if cfg_model is not None:
        for item in cfg_model:
            for layer_name in get_match_layers(item.layers, named_modules):
                layer = named_modules[layer_name]
                layer.requires_grad_(True)
                layer.train()
                train_params.append({'params':list(LoraBlock.extract_param_without_lora(layer).values()), 'lr':dict_get(item, 'lr', default_lr)})

    if cfg_lora is not None:
        for item in cfg_lora:
            for layer_name in get_match_layers(item.layers, named_modules):
                layer = named_modules[layer_name]
                arg_dict = {k:v for k,v in item.items() if k!='layers'}
                lora_block_dict = lora_layers.layer_map[getattr(arg_dict, 'type', 'lora')].warp_model(layer, **arg_dict)
                block_branch = getattr(item, 'branch', None) # for DreamArtist-lora
                for k,v in lora_block_dict.items():
                    if block_branch is None:
                        all_lora_blocks[f'{layer_name}.{k}'] = v
                    elif block_branch=='p':
                        all_lora_blocks[f'{layer_name}.{k}'] = v
                        v.set_mask((0.5, 1))
                    elif block_branch == 'n':
                        all_lora_blocks_neg[f'{layer_name}.{k}']=v
                        v.set_mask((0, 0.5))

                params_group=[]
                for block in lora_block_dict.values():
                    block.requires_grad_(True)
                    block.train()
                    params_group.extend(block.parameters())
                train_params.append({'params': params_group, 'lr':dict_get(item, 'lr', default_lr)})

    if len(all_lora_blocks_neg)>0:
        return train_params, (LoraGroup(all_lora_blocks), LoraGroup(all_lora_blocks_neg))
    else:
        return train_params, LoraGroup(all_lora_blocks)

def make_plugin(model, cfg_plugin, default_lr=1e-5):
    named_modules = {k:v for k,v in model.named_modules()}

    train_params=[]
    all_plugin_blocks={}

    # builder: functools.partial
    for plugin_name, builder in cfg_plugin:
        lr = getattr(builder.keywords, 'lr', default_lr)
        if 'lr' in builder.keywords:
            del builder.keywords['lr']
        plugin_class = getattr(builder.func, '__self__', builder.func) # support static or class method

        if issubclass(plugin_class, MultiPluginBlock):
            from_layers = [named_modules[item] for item in get_match_layers(builder.keywords['from_layers'], named_modules)]
            to_layers = [named_modules[item] for item in get_match_layers(builder.keywords['to_layers'], named_modules)]
            del builder.keywords['from_layers']
            del builder.keywords['to_layers']

            layer = builder(host_model=model, from_layers=from_layers, to_layers=to_layers)
            layer.requires_grad_(True)
            layer.train()
            setattr(model, plugin_name, layer)
            train_params.append({'params': layer.parameters(), 'lr': lr})
            all_plugin_blocks[plugin_name] = layer
        elif issubclass(plugin_class, SinglePluginBlock):
            layers_name = builder.keywords['layers']
            del builder.keywords['layers']
            for layer_name in get_match_layers(layers_name, named_modules):
                layer = builder(host_model=model, host=named_modules[layer_name])
                layer.requires_grad_(True)
                layer.train()
                setattr(named_modules[layer_name], plugin_name, layer)
                train_params.append({'params': layer.parameters(), 'lr': lr})
                all_plugin_blocks[f'{layer_name}.{plugin_name}'] = layer
        elif issubclass(plugin_class, PluginBlock):
            from_layer_name = builder.keywords['from_layer']
            from_layer = named_modules[from_layer_name]
            to_layer = named_modules[builder.keywords['to_layer']]
            del builder.keywords['from_layer']
            del builder.keywords['to_layer']

            layer = builder(host_model=model, from_layer=from_layer, to_layer=to_layer)
            layer.requires_grad_(True)
            layer.train()
            setattr(from_layer, plugin_name, layer)
            train_params.append({'params': layer.parameters(), 'lr': lr})
            all_plugin_blocks[f'{from_layer_name}.{plugin_name}'] = layer
        else:
            raise NotImplementedError(f'Unknown plugin {plugin_class}')
    return train_params, PluginGroup(all_plugin_blocks)

@torch.no_grad()
def load_hcpdiff(model:nn.Module, cfg_merge):
    named_modules = {k: v for k, v in model.named_modules()}
    named_params = {k: v for k, v in model.named_parameters()}
    all_lora_blocks = {}

    ckpt_manager_torch = CkptManagerPKL()
    ckpt_manager_safe = CkptManagerSafe()

    def get_ckpt_manager(path:str):
        return ckpt_manager_safe if path.endswith('.safetensors') else ckpt_manager_torch

    if "lora" in cfg_merge and cfg_merge.lora is not None:
        for item in cfg_merge.lora:
            lora_state = get_ckpt_manager(item.path).load_ckpt(item.path, map_location='cpu')['lora']
            lora_block_state = {}
            # get all layers in the lora_state
            for name, p in lora_state.items():
                lbidx = name.rfind('lora_block.')
                if lbidx != -1:
                    prefix = name[:lbidx - 1]
                    if prefix not in lora_block_state:
                        lora_block_state[prefix] = {}
                    lora_block_state[prefix][name[lbidx + len('lora_block.'):]] = p
            # get selected layers
            if item.layers != 'all':
                match_blocks = get_match_layers(item.layers, named_modules)
                match_layers = get_layers_with_block(named_modules, match_blocks, [nn.Linear, nn.Conv2d])
                lora_block_state = {k: v for k, v in lora_block_state.items() if k in match_layers}
            # add lora to host and load weights
            for host_name, lora_state in lora_block_state.items():
                rank_groups=dict_get(lora_state, 'layer.rank_groups', 1)
                if rank_groups>1:
                    if len(lora_state['layer.lora_down.weight'].shape)==3:
                        rank = rank_groups*lora_state['layer.lora_down.weight'].shape[2]
                    else:
                        rank = lora_state['layer.lora_down.weight'].shape[0]
                    lora_layer = lora_layers.layer_map['loha_group']
                else:
                    rank = lora_state['layer.lora_down.weight'].shape[0]
                    lora_layer = lora_layers.layer_map['lora']
                del lora_state['scale']

                lora_block_dict = lora_layer.warp_model(named_modules[host_name], rank=rank, dropout=dict_get(item, 'dropout', 0.0),
                                                 scale=dict_get(item, 'alpha', 1.0), bias='layer.lora_up.bias' in lora_state,
                                                 rank_groups=rank_groups)
                all_lora_blocks[f'{host_name}.lora_block'] = lora_block_dict['lora_block']
                lora_block_dict['lora_block'].load_state_dict(lora_state, strict=False)
                lora_block_dict['lora_block'].set_mask(dict_get(item, 'mask', None))
                lora_block_dict['lora_block'].to(model.device)

    if "part" in cfg_merge and cfg_merge.part is not None:
        for item in cfg_merge.part:
            part_state = get_ckpt_manager(item.path).load_ckpt(item.path, map_location='cpu')['base']
            if item.layers == 'all':
                for k, v in part_state.items():
                    named_params[k].data = cfg_merge.base_model_alpha * named_params[k].data + item.alpha * v
            else:
                match_blocks = get_match_layers(item.layers, named_modules)
                state_add = {k:v for blk in match_blocks for k,v in part_state.items() if k.startswith(blk)}
                for k, v in state_add.items():
                    named_params[k].data = cfg_merge.base_model_alpha * named_params[k].data + item.alpha * v

    if "plugin" in cfg_merge and cfg_merge.plugin is not None:
        for name, item in cfg_merge.plugin.items():
            plugin_state = get_ckpt_manager(item.path).load_ckpt(item.path, map_location='cpu')
            if item.layers == 'all':
                model.load_state_dict(plugin_state)
            else:
                match_blocks = get_match_layers(item.layers, named_modules)
                state_add = {k: v for blk in match_blocks for k, v in plugin_state.items() if k.startswith(blk)}
                model.load_state_dict(state_add)
            del item.layers
            del item.path
            getattr(model, name).set_hyper_params(**item)

    return LoraGroup(all_lora_blocks)