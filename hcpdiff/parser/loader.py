from hcpdiff.models.lora_layers_patch import LoraLayer
from rainbowneko.ckpt_manager import CkptManagerBase
from torch import nn
from hcpdiff.utils.net_utils import split_module_name
from rainbowneko.parser.model import NekoPluginLoader
from rainbowneko.parser.model.locator import get_match_layers

def get_lora_rank_and_cls(lora_state):
    if 'layer.W_down' in lora_state:
        rank = lora_state['layer.W_down'].shape[0]
        return LoraLayer, rank
    else:
        raise ValueError('Unknown lora format.')

class HCPLoraLoader(NekoPluginLoader):

    def load_to(self, name, model):
        # get model to load plugin and its named_modules
        model = model if self.module_to_load == '' else eval(f"model.{self.module_to_load}")

        named_modules = {k:v for k, v in model.named_modules()}
        plugin_state = self.ckpt_manager.load(self.path, map_location='cpu')['plugin_ema' if self.load_ema else 'plugin']

        # filter layers to load
        if self.layers != 'all':
            match_blocks = get_match_layers(self.layers, named_modules)
            plugin_state = {k: v for blk in match_blocks for k, v in plugin_state.items() if k.startswith(blk)}

        if self.state_prefix:
            state_prefix_len = len(self.state_prefix)
            plugin_state = {k[state_prefix_len:]: v for k, v in plugin_state.items() if k.startswith(self.state_prefix)}

        lora_block_state = {}
        # get all layers in the lora_state
        for pname, p in plugin_state.items():
            # lora_block. is the old format
            prefix, block_name = pname.split('.___.', 1)
            if prefix not in lora_block_state:
                lora_block_state[prefix] = {}
            lora_block_state[prefix][block_name] = p

        # add lora to host and load weights
        for layer_name, lora_state in lora_block_state.items():
            lora_layer_cls, rank = get_lora_rank_and_cls(lora_state)

            if 'alpha' in lora_state:
                del lora_state['alpha']

            parent_name, host_name = split_module_name(layer_name)

            lora_block = lora_layer_cls.wrap_layer(name, named_modules[layer_name], rank=rank, bias='layer.bias' in lora_state,
                                                parent_block=named_modules[parent_name], host_name=host_name)
            lora_block.set_hyper_params(**self.plugin_kwargs)

        # Load state to plugin
        plugin_state = {k.replace('___', name): v for k, v in plugin_state.items()}  # replace placeholder to target plugin name
        load_info = model.load_state_dict(plugin_state, strict=False)
        if len(load_info.unexpected_keys) > 0:
            print(name, 'unexpected_keys', load_info.unexpected_keys)