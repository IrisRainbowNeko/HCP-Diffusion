"""
plugin.py
====================
    :Name:        model plugin
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from typing import Tuple, List, Dict, Any

import torch
from torch import nn
import weakref

from hcpdiff.utils.utils import isinstance_list

class BasePluginBlock(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, host:nn.Module, fea_in:Tuple[torch.Tensor], fea_out:torch.Tensor):
        return fea_out

    def remove(self):
        pass

    def feed_input_data(self, data):
        self.input_data = data

    def register_input_feeder_to(self, host_model):
        if not hasattr(host_model, 'input_feeder'):
            host_model.input_feeder = []
        host_model.input_feeder.append(self.feed_input_data)

    def set_hyper_params(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def extract_state_without_plugin(model: nn.Module, trainable=False):
        trainable_keys = {k for k, v in model.named_parameters() if v.requires_grad}
        plugin_names = {k for k,v in model.named_modules() if isinstance(v, BasePluginBlock)}
        model_sd = {}
        for k, v in model.state_dict().items():
            if (not trainable) or k in trainable_keys:
                for name in plugin_names:
                    if k.startswith(name):
                        break
                else:
                    model_sd[k]=v
        return model_sd

class SinglePluginBlock(BasePluginBlock):
    wrapable_classes=[]

    def __init__(self, name:str, host:nn.Module, hook_param=None, host_model=None):
        super().__init__(name)
        self.host = weakref.ref(host)
        setattr(host, name, self)

        if hook_param is None:
            self.hook_handle = host.register_forward_hook(self.layer_hook)
        else: # hook for model parameters
            self.backup = getattr(host, hook_param)
            self.target = hook_param
            self.handle_pre = host.register_forward_pre_hook(self.pre_hook)
            self.handle_post = host.register_forward_hook(self.post_hook)

    def layer_hook(self, host, fea_in:Tuple[torch.Tensor], fea_out:torch.Tensor):
        return self(fea_in, fea_out)

    def pre_hook(self, host, fea_in:torch.Tensor):
        host.weight_restored = False
        host_param = getattr(host, self.target)
        delattr(host, self.target)
        setattr(host, self.target, self(host_param))
        return fea_in

    def post_hook(self, host, fea_int, fea_out):
        if not getattr(host, 'weight_restored', False):
            setattr(host, self.target, self.backup)
            host.weight_restored = True

    def remove(self):
        host = self.host()
        delattr(host, self.name)
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()
        else:
            self.handle_pre.remove()
            self.handle_post.remove()

    @classmethod
    def wrap_layer(cls, name: str, layer: nn.Module, **kwargs):  # -> SinglePluginBlock:
        raise NotImplementedError(f'wrap_layer of {cls} is not implemented.')

    @classmethod
    def wrap_model(cls, plugin_name: str, model: nn.Module, exclude_key=None, **kwargs):  # -> Dict[str, SinglePluginBlock]:
        plugin_block_dict = {}
        if isinstance_list(model, cls.wrapable_classes):
            plugin_block_dict[''] = cls.wrap_layer(plugin_name, model, **kwargs)
        else:
            if exclude_key:
                # there maybe multiple single plugin block, avoid insert plugin into plugin blocks with exclude_key
                named_modules = {name: layer for name, layer in model.named_modules() if exclude_key not in name}
            else:
                named_modules = {name: layer for name, layer in model.named_modules()}
            for name, layer in named_modules.items():
                if isinstance_list(layer, cls.wrapable_classes):
                    plugin_block_dict[name] = cls.wrap_layer(plugin_name, layer, **kwargs)
        return plugin_block_dict

class PluginBlock(BasePluginBlock):
    def __init__(self, name, from_layer:Dict[str, Any], to_layer:Dict[str, Any], host_model=None):
        super().__init__(name)
        self.host_from = weakref.ref(from_layer['layer'])
        self.host_to = weakref.ref(to_layer['layer'])
        setattr(from_layer['layer'], name, self)

        if from_layer['pre_hook']:
            self.hook_handle_from = from_layer['layer'].register_forward_pre_hook(lambda host, fea_in: self.from_layer_hook(host, fea_in, None))
        else:
            self.hook_handle_from = from_layer['layer'].register_forward_hook(lambda host, fea_in, fea_out: self.from_layer_hook(host, fea_in, fea_out))
        if to_layer['pre_hook']:
            self.hook_handle_to = to_layer['layer'].register_forward_pre_hook(lambda host, fea_in: self.to_layer_hook(host, fea_in, None))
        else:
            self.hook_handle_to = to_layer['layer'].register_forward_hook(lambda host, fea_in, fea_out: self.to_layer_hook(host, fea_in, fea_out))

    def from_layer_hook(self, host, fea_in:Tuple[torch.Tensor], fea_out:torch.Tensor):
        self.feat_from = fea_in

    def to_layer_hook(self, host, fea_in:Tuple[torch.Tensor], fea_out:torch.Tensor):
        return self(self.feat_from, fea_in, fea_out)

    def remove(self):
        host_from = self.host_from()
        delattr(host_from, self.name)
        self.hook_handle_from.remove()
        self.hook_handle_to.remove()

class MultiPluginBlock(BasePluginBlock):
    def __init__(self, name:str, from_layers:List[Dict[str, Any]], to_layers:List[Dict[str, Any]], host_model=None):
        super().__init__(name)
        assert host_model is not None
        self.host_from = [weakref.ref(x['layer']) for x in from_layers]
        self.host_to = [weakref.ref(x['layer']) for x in to_layers]
        self.host_model = weakref.ref(host_model)
        setattr(host_model, name, self)

        self.feat_from=[None for _ in range(len(from_layers))]

        self.hook_handle_from = []
        self.hook_handle_to = []

        for idx, layer in enumerate(from_layers):
            if layer['pre_hook']:
                handle_from = layer['layer'].register_forward_pre_hook(lambda host, fea_in, idx=idx:self.from_layer_hook(host, fea_in, None, idx))
            else:
                handle_from = layer['layer'].register_forward_hook(lambda host, fea_in, fea_out, idx=idx: self.from_layer_hook(host, fea_in, fea_out, idx))
            self.hook_handle_from.append(handle_from)
        for idx, layer in enumerate(to_layers):
            if layer['pre_hook']:
                handle_to = layer['layer'].register_forward_pre_hook(lambda host, fea_in, idx=idx:self.to_layer_hook(host, fea_in, None, idx))
            else:
                handle_to = layer['layer'].register_forward_hook(lambda host, fea_in, fea_out, idx=idx:self.to_layer_hook(host, fea_in, fea_out, idx))
            self.hook_handle_to.append(handle_to)

        self.record_count=0

    def from_layer_hook(self, host, fea_in:Tuple[torch.Tensor], fea_out:Tuple[torch.Tensor], idx: int):
        self.feat_from[idx] = fea_in
        self.record_count+=1
        if self.record_count==len(self.feat_from): # call forward when all feat is record
            self.record_count = 0
            self.feat_to = self(self.feat_from)

    def to_layer_hook(self, host, fea_in:Tuple[torch.Tensor], fea_out:Tuple[torch.Tensor], idx: int):
        return self.feat_to[idx] + fea_out

    def remove(self):
        host_model = self.host_model()
        delattr(host_model, self.name)
        for handle_from in self.hook_handle_from:
            handle_from.remove()
        for handle_to in self.hook_handle_to:
            handle_to.remove()

class WrapPluginContainer(nn.Module):
    def __init__(self, plugin, name, host):
        super().__init__()
        self.name = name
        self._host = host
        super(WrapPluginContainer, self).__setattr__(name, plugin)

    def forward(self, *args, **kwargs):
        return getattr(self, self.name)(*args, **kwargs)

    def __getattr__(self, item):
        try:
            return super(WrapPluginContainer, self).__getattr__(item)
        except:
            return getattr(self._host, item)

    def __setattr__(self, key, value):
        if key == '_host' or key == 'name' or key == self.name:
            super(WrapPluginContainer, self).__setattr__(key, value)
        else:
            setattr(self._host, key, value)


class WrapPluginBlock(BasePluginBlock):
    def __init__(self, name:str, host:nn.Module, host_model=None, parent_block:nn.Module=None, host_name:str=None):
        super().__init__(name)
        self.host = weakref.ref(host)
        self.parent_block = weakref.ref(parent_block)
        self.host_name = host_name

        container = WrapPluginContainer(self, name, host)

        delattr(parent_block, host_name)
        #setattr(parent_block, f'{host_name}_origin_block', host)
        setattr(parent_block, host_name, container)

    def forward(self, *args, **kwargs):
        args, kwargs = self.pre_forward(*args, **kwargs)
        output = self.host()(*args, **kwargs)
        output = self.post_forward(output, *args, **kwargs)
        return output

    def pre_forward(self, *args, **kwargs):
        return args, kwargs

    def post_forward(self, output, *args, **kwargs):
        return output

    def remove(self):
        parent_block = self.parent_block()
        delattr(parent_block, self.host_name)
        #delattr(parent_block, f'{self.host_name}_origin_block')
        setattr(parent_block, self.host_name, self.host())

class PluginGroup:
    def __init__(self, plugin_dict:Dict[str, BasePluginBlock]):
        self.plugin_dict = plugin_dict # {host_model_path: plugin_object}

    def __setitem__(self, k, v):
        self.plugin_dict[k]=v

    def __getitem__(self, k):
        return self.plugin_dict[k]

    @property
    def plugin_name(self):
        if self.empty():
            return None
        return next(iter(self.plugin_dict.values())).name

    def remove(self):
        for plugin in self.plugin_dict.values():
            plugin.remove()

    def state_dict(self, model=None):
        if model is None:
            return {f'{k}.___.{ks}':vs for k,v in self.plugin_dict.items() for ks,vs in v.state_dict().items()}
        else:
            sd_model = model.state_dict()
            return {f'{k}.___.{ks}':sd_model[f'{k}.{v.name}.{ks}'] for k,v in self.plugin_dict.items() for ks,vs in v.state_dict().items()}

    def empty(self):
        return len(self.plugin_dict)==0

