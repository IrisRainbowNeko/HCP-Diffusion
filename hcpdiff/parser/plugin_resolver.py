from hcpdiff.models.lora_layers_patch import LoraLayer
from rainbowneko.ckpt_manager import CkptManagerBase
from torch import nn
from hcpdiff.utils.net_utils import split_module_name

def get_lora_rank_and_cls(lora_state):
    if 'layer.W_down' in lora_state:
        rank = lora_state['layer.W_down'].shape[0]
        return LoraLayer, rank
    else:
        raise ValueError('Unknown lora format.')

def lora_resolver(lora_name:str, model:nn.Module, path:str, ckpt_manager:CkptManagerBase, load_ema: bool):
    named_modules = {k:v for k, v in model.named_modules()}
    lora_state = ckpt_manager.load(path, map_location='cpu')['lora_ema' if load_ema else 'lora']
    lora_block_state = {}
    # get all layers in the lora_state
    for name, p in lora_state.items():
        # lora_block. is the old format
        prefix, block_name = name.split('.___.' if name.rfind('lora_block.') == -1 else '.lora_block.', 1)
        if prefix not in lora_block_state:
            lora_block_state[prefix] = {}
        lora_block_state[prefix][block_name] = p

    # add lora to host and load weights
    for layer_name, lora_state in lora_block_state.items():
        lora_layer_cls, rank = get_lora_rank_and_cls(lora_state)

        if 'alpha' in lora_state:
            del lora_state['alpha']

        parent_name, host_name = split_module_name(layer_name)

        lora_block = lora_layer_cls.wrap_layer(lora_name, named_modules[layer_name], rank=rank, bias='layer.bias' in lora_state,
                                               parent_block=named_modules[parent_name], host_name=host_name)