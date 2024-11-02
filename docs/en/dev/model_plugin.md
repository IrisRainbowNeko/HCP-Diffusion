# Model Plugin Development Guide

Model plugins are defined in `hcpdiff.models.plugin` and have several plugin types, including:

| Plugin Type       | Regular expression for batch attach(re prefix) | prehook prefix |
|-------------------|:----------------------------------------------:|:--------------:|
| SinglePluginBlock |                    &#10004;                    |    &#10006;    |
| PluginBlock       |                    &#10004;                    |    &#10004;    |
| MultiPluginBlock  |                    &#10006;                    |    &#10004;    |
| PatchPluginBlock  |                    &#10004;                    |    &#10006;    |

## SinglePluginBlock
`SinglePluginBlock` can be attached to any layer of the model and can receive both the input and output results of its host layer in the `forward` function. For example, the `SinglePluginBlock` can be used to implement a LoRA plugin:
```python
class LoraBlock(SinglePluginBlock):
    wrapable_classes = [nn.Linear, nn.Conv2d]

    def __init__(self, lora_id:int, host:Union[nn.Linear, nn.Conv2d], rank, dropout=0.1, alpha=1.0, bias=False,
                 inplace=True, hook_param=None, alpha_auto_scale=True, **kwargs):
        super().__init__(f'lora_block_{lora_id}', host, hook_param)

        self.mask_range = None
        self.inplace=inplace
        self.bias=bias

        if isinstance(host, nn.Linear):
            self.host_type = 'linear'
            self.layer = self.LinearLayer(host, rank, bias, dropout, self)
        elif isinstance(host, nn.Conv2d):
            self.host_type = 'conv'
            self.layer = self.Conv2dLayer(host, rank, bias, dropout, self)
        else:
            raise NotImplementedError(f'No lora for {type(host)}')
        self.rank = self.layer.rank

        self.register_buffer('alpha', torch.tensor(alpha/self.rank if alpha_auto_scale else alpha))

        def forward(self, fea_in:Tuple[torch.Tensor], fea_out:torch.Tensor):
            return fea_out + self.layer(fea_in[0]) * self.alpha
```

The class attribute `wrapable_classes` defined here can be automatically attached to all sub-modules of the specified module that is instance of the classes defined in `wrapable_classes` when calling `LoraBlock.wrap_model`.

### Usage in Configuration Files

In configuration files, you can use Hydra's syntax rules with `_target_` to create a plugin by its class path. `alpha` is a parameter declared in the `LoraBlock` class, and you can configure any additional parameters in a similar way.

The `layers` attribute corresponds to `SinglePluginBlock` and is used to specify which layers the plugin should be attached to. You can use regular expressions (prefixed with `re:`) to select multiple layers in bulk.

```yaml
plugin_unet:
  lora1:
    _target_: model_plugin.LoraBlock
    _partial_: True # Required
    lr: 1e-4 # Learning rate for this module
    alpha: 0.5 # Parameter declared in the LoraBlock module
    layers: # Specify which layers to attach the plugin
      - 're:.*\.attn.?$'
      - 're:.*\.ff$'
```

### Hooking Model Parameters

In addition to supporting direct modification of output results, `SinglePluginBlock` can also modify parameters of the host layer. You just need to specify the parameters you want to modify using `hook_param`. In this case, the input to `forward` is the original parameters specified by `hook_param`, and the output result will be used as the new parameters. For example, to implement a `loha` plugin:

```python
class LohaLayer(LoraBlock):
    def __init__(self, lora_id:int, host, rank=1, dropout=0.1, alpha=1.0, bias=False, inplace=True, rank_groups=2, alpha_auto_scale=True, **kwargs):
        self.rank_groups_raw = rank_groups
        super().__init__(lora_id, host, rank, dropout, alpha, bias, inplace, hook_param='weight', alpha_auto_scale=alpha_auto_scale)

    def forward(self, host_param: nn.Parameter):
        return host_param + self.layer(host_param) * self.alpha
```

## PluginBlock

`PluginBlock` allows you to select an input layer and an output layer, and in the `forward` function, it can receive the input or output results from these layers and modify the input or output results of the output layer. For example, you can use `PluginBlock` to implement residual connections:

```python
class Skip(PluginBlock):

    def __init__(self, name, from_layer:Dict[str, Any], to_layer:Dict[str, Any], **kwargs):
        super().__init__(name, from_layer, to_layer)
        self.layer = nn.Linear(from_layer['layer'].in_features, to_layer['layer'].out_features)

    def forward(self, fea_from_in:Tuple[torch.Tensor], fea_in:Tuple[torch.Tensor], fea_out:torch.Tensor):
        return fea_out + self.layer(fea_from_in[0])
```

### Usage in Configuration Files

The usage is similar to `SinglePluginBlock`.

`from_layer` and `to_layer` represent the input and output layers, respectively. You can use regular expressions to specify multiple layers for attachment. However, the number of `from_layer` and `to_layer` entries must match and be a pair.

You can add the `prehook:` prefix to call `register_forward_pre_hook`, which allows you to modify the input variables of the output layer.

```yaml
plugin_unet:
  loha1:
    _target_: model_plugin.Skip
    _partial_: True
    lr: 1e-4

    from_layer:
      - 're:.*proj_in$'
    to_layer:
      - 'prehook:re:.*proj_out$'
```

## MultiPluginBlock

`MultiPluginBlock` allows you to select multiple input layers and multiple output layers. In the `forward` function, it can receive the input or output from the input layers and modify the input or output of the output layers. For example, you can use `MultiPluginBlock` to implement a controlnet:

```python
class ControlNetPlugin(MultiPluginBlock):

    def __init__(self, name:str, from_layers: List[Dict[str, Any]], to_layers: List[Dict[str, Any]], host_model: UNet2DConditionModel=None,
                 cond_block_channels=(3, 16, 32, 96, 256, 320),
                 layers_per_block=2, block_out_channels: Tuple[int] = (320, 640, 1280, 1280)):
        super().__init__(name, from_layers, to_layers, host_model=host_model)

        # recive the condition image
        self.register_input_feeder_to(host_model)

        # Define model 
        ...

    # Rewriting Input and Output Data Processing for ControlNet
    def from_layer_hook(self, host, fea_in:Tuple[torch.Tensor], fea_out:Tuple[torch.Tensor], idx: int):
        if idx==0:
            self.data_input = fea_in
        elif idx==1:
            self.feat_to = self(*self.data_input)

    def to_layer_hook(self, host, fea_in:Tuple[torch.Tensor], fea_out:Tuple[torch.Tensor], idx: int):
        if idx == 5:
            sp = fea_in[0].shape[1]//2
            new_feat = fea_in[0].clone()
            new_feat[:, sp:, ...] = fea_in[0][:, sp:, ...] + self.feat_to[0]
            return (new_feat, fea_in[1])
        elif idx == 3:
            return (fea_out[0], tuple(fea_out[1][i] + self.feat_to[(idx) * 3 + i+1] for i in range(2)))
        elif idx == 4:
            return fea_out + self.feat_to[11+1]
        else:
            return (fea_out[0], tuple(fea_out[1][i]+self.feat_to[(idx)*3+i+1] for i in range(3)))

    # ControlNet forward
    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple:
        ...
        return controlnet_down_block_res_samples + (mid_block_res_sample,)
```

### Usage in Configuration Files

The usage is similar to `PluginBlock`.

`from_layers` and `to_layers` represent the input and output layers, respectively.

You can also use regular expressions here, but the meaning is different from other plugins. When using regular expressions here, you can define multiple input and output layers for a plugin in bulk.

You can add the `prehook:` prefix to attach to the input part.

```yaml
plugin_unet:
  controlnet1:
    _target_: hcpdiff.models.controlnet.ControlNetPlugin
    _partial_: True
    lr: 1e-4
    from_layers:
      - 'pre_hook:' # get model input
      - 'pre_hook:conv_in' # make forward in the autocast context
    to_layers:
      - 'down_blocks.0'
      - 'down_blocks.1'
      - 'down_blocks.2'
      - 'down_blocks.3'
      - 'mid_block'
      - 'pre_hook:up_blocks.3.resnets.2'
```

## PatchPluginBlock

The usage of `PatchPluginBlock` is similar to `SinglePluginBlock`. It is mainly used to completely replace a module in the original model, such as replacing a Transformer module with a convolution module or modifying the `property` of the original module.

When attaching a `PatchPluginBlock` to a module, the original module will be replaced with the corresponding `PatchPluginContainer` obtained through `PatchPluginBlock.get_container`, and the `PatchPluginBlock` to be attached will become an attribute of this `PatchPluginContainer`. A `PatchPluginContainer` allows you to add multiple `PatchPluginBlocks`. If the module has already been patched and has become a `PatchPluginContainer`, the current `PatchPluginBlock` will be added directly to this `PatchPluginContainer`.

For example, replacing `SiLU` with `PReLU`:
```python
class PReLUPatchContainer(PatchPluginContainer):
    def forward(self, *args, **kwargs):
        output = 0.0
        for name in self.plugin_names:
            output = getattr(self, name).post_forward(output, *args, **kwargs)
        return output

class PReLUPatch(PatchPluginBlock):
    container_cls = PReLUPatchContainer
    def __init__(self, *args, p=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.act = nn.PReLU(p)

    def post_forward(self, output, x):
        return output + self.act(x)
```

### Usage in Configuration Files

The usage is similar to `SinglePluginBlock`:
```yaml
plugin_unet:
  prelu:
    _target_: model_plugin.PReLUPatch
    _partial_: True
    lr: 1e-4
    layers:
      - 're:.*nonlinearity$'
```

## Accessing Additional Inputs (Extra Control Conditions, Timesteps, etc.)

All the contents returned in `plugin_input` from `load_data` in the `Dataset` will be fed into the `input_feeder` of the `plugin`. This allows the `plugin` to access additional inputs.

First, define an `input_feeder` to retrieve the image:
```python
def feed_input_data(self, data): # get the condition image
    if isinstance(data, dict):
        self.cond = data['cond']
```

Then, register the `input_feeder` in the host model:
```python
self.register_input_feeder_to(host_model)
```

This way, you can access additional input information during model computations. The `input_feeder` mechanism is supported by all types of plugins.

For the specific implementation of `input_feeder`, refer to `models.wrapper`.