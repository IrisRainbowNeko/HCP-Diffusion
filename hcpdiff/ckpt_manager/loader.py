from hcpdiff.models.lora_layers_patch import LoraLayer
from torch import nn
from hcpdiff.utils.net_utils import split_module_name
from rainbowneko.ckpt_manager import NekoPluginLoader, LocalCkptSource, CkptFormat
from rainbowneko.ckpt_manager.locator import get_match_layers
from rainbowneko.models.plugin import PluginGroup

def get_lora_rank_and_cls(lora_state):
    if 'layer.W_down' in lora_state:
        rank = lora_state['layer.W_down'].shape[0]
        return LoraLayer, rank
    else:
        raise ValueError('Unknown lora format.')

class HCPLoraLoader(NekoPluginLoader):
    def __init__(self, format: CkptFormat=None, source: LocalCkptSource=None, path: str = None, layers='all', target_plugin=None,
                 state_prefix=None, base_model_alpha=0.0, load_ema=False, module_to_load='', key_map=None, **plugin_kwargs):
        key_map = key_map or ('name -> name', 'model -> model')
        super().__init__(format, source, path=path, layers=layers, target_plugin=target_plugin, state_prefix=state_prefix,
                         base_model_alpha=base_model_alpha, load_ema=load_ema, key_map=key_map, **plugin_kwargs)
        self.module_to_load = module_to_load

    def _load_to(self, name, model):
        # get model to load plugin and its named_modules
        model = model if self.module_to_load == '' else eval(f"model.{self.module_to_load}")

        named_modules = {k:v for k, v in model.named_modules()}
        state_dict = self.load(self.path, map_location='cpu')
        if 'base' in state_dict or 'base_ema' in state_dict:
            plugin_state = state_dict['base_ema' if self.load_ema else 'base']
        elif 'plugin' in state_dict or 'plugin_ema' in state_dict:
            plugin_state = state_dict['plugin_ema' if self.load_ema else 'plugin']
        else:
            plugin_state = state_dict

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
        lora_blocks = {}
        for layer_name, lora_state in lora_block_state.items():
            lora_layer_cls, rank = get_lora_rank_and_cls(lora_state)

            if 'alpha' in lora_state:
                lora_state['alpha'] *= self.plugin_kwargs.get('alpha', 1.0)

            parent_name, host_name = split_module_name(layer_name)

            lora_block = lora_layer_cls.wrap_layer(name, named_modules[layer_name], rank=rank, bias='layer.bias' in lora_state,
                                                parent_block=named_modules[parent_name], host_name=host_name)
            lora_block.set_hyper_params(**self.plugin_kwargs)
            lora_blocks[layer_name] = lora_block
            load_info = lora_block.load_state_dict(lora_state, strict=False)
            if len(load_info.unexpected_keys) > 0:
                print(name, 'unexpected_keys', load_info.unexpected_keys)
        return PluginGroup(lora_blocks)