# 模型插件开发指南

模型插件定义在`hcpdiff.models.plugin`中，包含以下几种基本类型:

| 插件类型            | 正则表达式批量插入(re前缀)  | prehook前缀 |
|-------------------|:-----------------:|:-----------:|
| SinglePluginBlock |     &#10004;      |   &#10006;  |
| PluginBlock       |     &#10004;      |   &#10004;  |
| MultiPluginBlock  |     &#10006;      |   &#10004;  |
| PatchPluginBlock  |     &#10004;      |   &#10006;  |

## SinglePluginBlock
`SinglePluginBlock`可以挂载到模型的任意一层，在`forward`函数中可以接收其宿主层的输入与输出结果。
比如用`SinglePluginBlock`可以实现lora插件:
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

其中定义的类属性`wrapable_classes`，可以在调用`LoraBlock.wrap_model`时，自动挂载到指定模块的所有满足`wrapable_classes`的子模块上。

### 在配置文件中使用

在配置文件中使用`hydra`的语法规则，`_target_`填写插件类的路径。`alpha`是LoraBlock类声明的参数，所有额外的参数都可以通过类似的方式配置。

`layers`是`SinglePluginBlock`对应的属性，配置将插件添加到哪些层上，可以使用正则表达式(`re:`前缀)批量选取。

```yaml
plugin_unet:
  lora1:
    _target_: model_plugin.LoraBlock
    _partial_: True # 必须加
    lr: 1e-4 # 这部分模块的学习率
    alpha: 0.5 # LoraBlock模块声明的参数
    layers: # 添加到哪些层上
      - 're:.*\.attn.?$'
      - 're:.*\.ff$'
```

### hook模型参数
`SinglePluginBlock`除了支持直接修改输出结果，还可以修改宿主层的参数。只需要通过`hook_param`指定需要修改的参数即可。这种情况，`forward`输入的是`hook_param`指定的原有参数，输出的结果会作为新的参数。例如实现`loha`插件:

```python
class LohaLayer(LoraBlock):
    def __init__(self, lora_id:int, host, rank=1, dropout=0.1, alpha=1.0, bias=False, inplace=True, rank_groups=2, alpha_auto_scale=True, **kwargs):
        self.rank_groups_raw = rank_groups
        super().__init__(lora_id, host, rank, dropout, alpha, bias, inplace, hook_param='weight', alpha_auto_scale=alpha_auto_scale)

    def forward(self, host_param: nn.Parameter):
        return host_param + self.layer(host_param) * self.alpha
```

## PluginBlock

`PluginBlock`可以选取一个输入层和一个输出层，在`forward`函数中可以接收输入和输出层的输入或输出结果，并改变输出层的输入或输出结果。
比如用`SinglePluginBlock`可以实现残差连接:

```python
class Skip(PluginBlock):

    def __init__(self, name, from_layer:Dict[str, Any], to_layer:Dict[str, Any], **kwargs):
        super().__init__(name, from_layer, to_layer)
        self.layer = nn.Linear(from_layer['layer'].in_features, to_layer['layer'].out_features)

    def forward(self, fea_from_in:Tuple[torch.Tensor], fea_in:Tuple[torch.Tensor], fea_out:torch.Tensor):
        return fea_out + self.layer(fea_from_in[0])
```

### 在配置文件中使用

使用方式与`SinglePluginBlock`类似。
`from_layer`和`to_layer`分别是输入层和输出层，可以使用正则表达式批量挂载，但需要`from_layer`和`to_layer`数量相同，有配对关系。

可以添加`prehook:`前缀，这样会调用`register_forward_pre_hook`，可以修改输出层的输入变量。

```yaml
plugin_unet:
  loha1:
    _target_: model_plugin.Skip
    _partial_: True # 必须加
    lr: 1e-4 # 这部分模块的学习率

    from_layer:
      - 're:.*proj_in$'
    to_layer:
      - 'prehook:re:.*proj_out$'
```

## MultiPluginBlock

`MultiPluginBlock`可以选取多个输入层和多个输出层，在`forward`函数中可以接收输入层的输入或输出，并改变输出层的输入或输出。
比如用`MultiPluginBlock`可以实现controlnet:

```python
class ControlNetPlugin(MultiPluginBlock):

    def __init__(self, name:str, from_layers: List[Dict[str, Any]], to_layers: List[Dict[str, Any]], host_model: UNet2DConditionModel=None,
                 cond_block_channels=(3, 16, 32, 96, 256, 320),
                 layers_per_block=2, block_out_channels: Tuple[int] = (320, 640, 1280, 1280)):
        super().__init__(name, from_layers, to_layers, host_model=host_model)

        # 接受图片控制条件输入
        self.register_input_feeder_to(host_model)

        #模型定义
        ...

    #根据 ControlNet 特性，重写输入输出层数据处理方式
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

    # ControlNet正常forward
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

### 在配置文件中使用

使用方式与`PluginBlock`类似。
`from_layers`和`to_layers`分别是输入层和输出层。
这块也可以使用正则表达式定义，但含义和其他插件有区别。这里使用正则表达式，可以批量定义一个插件的多个输入和输出层。

可以添加`prehook:`前缀，可以挂载到输入部分。

```yaml
plugin_unet:
  controlnet1:
    _target_: hcpdiff.models.controlnet.ControlNetPlugin
    _partial_: True
    lr: 1e-4
    from_layers:
      - 'pre_hook:' #获取模型输入
      - 'pre_hook:conv_in' # 让forward在autocast范围内
    to_layers:
      - 'down_blocks.0'
      - 'down_blocks.1'
      - 'down_blocks.2'
      - 'down_blocks.3'
      - 'mid_block'
      - 'pre_hook:up_blocks.3.resnets.2'
```

## PatchPluginBlock

`PatchPluginBlock`用法与`SinglePluginBlock`类似。主要用于整个替换原有模型的模块，比如把一个Transformer模块换成卷积模块，或者修改原有模型的`property`。

为一个模块挂载`PatchPluginBlock`时，会把原有模块换成通过`PatchPluginBlock.get_container`得到的对应的`PatchPluginContainer`，而需要挂载的`PatchPluginBlock`会作为这个`PatchPluginContainer`的一个属性。一个`PatchPluginContainer`允许添加多个`PatchPluginBlock`进去。如果模块已经被patch，变成了`PatchPluginContainer`，那么就会在这个`PatchPluginContainer`中直接添加当前的`PatchPluginBlock`。

比如把`SiLU`换成`PReLU`:
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

### 在配置文件中使用

用法与`SinglePluginBlock`类似:
```yaml
plugin_unet:
  prelu:
    _target_: model_plugin.PReLUPatch
    _partial_: True
    lr: 1e-4
    layers:
      - 're:.*nonlinearity$'
```

### 自动挂载到支持的子模块上
定义`wrapable_classes`，指定可以挂载的模块:
```python
class PReLUPatch(PatchPluginBlock):
    wrapable_classes = (nn.SiLU,)
    ...
```

在配置文件中调用对应的`wrap_model`方法，自动挂载到子模块上:
```yaml
plugin_unet:
  prelu:
    _target_: model_plugin.PReLUPatch.wrap_model
    _partial_: True
    lr: 1e-4
    layers:
      - ''
```

> 如果你的Plugin中定义的子模块，有`wrapable_classes`中的类别，则需要为`wrap_model`添加`exclude_key`，避免挂载多个模块时出错。
> 
> 比如lora模块的层名称固定有`lora_block_`:
> ```
> @classmethod
> def wrap_model(cls, lora_id:int, model: nn.Module, **kwargs):# -> Dict[str, LoraBlock]:
>     return super(LoraBlock, cls).wrap_model(lora_id, model, exclude_key='lora_block_', **kwargs)
> ```


## 获取额外输入(额外控制条件，timesteps等)

在`Dataset`的`load_data`中返回的`plugin_input`中的所有内容都会被输入到`plugin`的`input_feeder`中。`plugin`可以通过这种方式获取额外输入。

首先定义`input_feeder`，用于获取图像:
```python
def feed_input_data(self, data): # get the condition image
    if isinstance(data, dict):
        self.cond = data['cond']
```

之后在宿主模型中注册`input_feeder`:
```python
self.register_input_feeder_to(host_model)
```

这样就可以在模型运算过程中获取额外的输入信息，所有类型的`plugin`都支持`input_feeder`机制。

`input_feeder`具体实现见`models.wrapper`