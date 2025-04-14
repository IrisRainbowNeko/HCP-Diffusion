# 模型文件格式说明

目前基础模型和lora模型都有多种不同的格式，不同的框架支持的格式不同。HCP-Diffusion可以在配置文件中选择读取和保存各种不同格式的模型文件。比如HCP-Diffusion的格式，diffusers格式，webui的ckpt格式，lora的webui格式等。

## 基础模型格式

目前的基础模型有```.ckpt```或```.safetensors```的单个文件格式(webui和comfyui支持的格式)，也有以文件夹形式保存各个组件的`diffusers`格式，这些格式都可以再配置文件中选择性的读取和保存。

::::{tab-set}
:::{tab-item} diffusers格式

**读取模型**

HCP-Diffusion提供了简化的`SD15_auto_loader`和`SDXL_auto_loader`可以自动识别给定路径是单个文件格式还是`diffusers`格式，简化配置:

```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.easy import SD15_auto_loader, SDXL_auto_loader

wrapper=SD15Wrapper.from_pretrained( # 加载预训练基础模型
    _partial_=True,
    # 自动识别格式并读取模型参数
    models=SD15_auto_loader(ckpt_path='Lykon/DreamShaper', _partial_=True),
),
```

````{note}
也可以通过更基础的API来读取模型，指定使用单文件格式:
```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.ckpt_manager import DiffusersSD15Format
from rainbowneko.ckpt_manager import LocalCkptSource, NekoLoader

wrapper=SD15Wrapper.from_pretrained(
    _partial_=True,
    models=NekoLoader(
        format=DiffusersSD15Format(), # 指定diffusers格式
        source=LocalCkptSource(), # 从路径读取 (可以是huggingface repo)
    ).load(path='Lykon/DreamShaper', _partial_=True)
),
```
````

**保存模型**

🚧 开发中

:::

:::{tab-item} 单文件格式

**读取模型**

HCP-Diffusion提供了简化的`SD15_auto_loader`和`SDXL_auto_loader`可以自动识别给定路径是单个文件格式还是`diffusers`格式，简化配置:

```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.easy import SD15_auto_loader, SDXL_auto_loader

wrapper=SD15Wrapper.from_pretrained( # 加载预训练基础模型
    _partial_=True,
    # 自动识别格式并读取模型参数
    models=SD15_auto_loader(ckpt_path='ckpt/DreamShaper.safetensors', _partial_=True),
),
```

````{note}
也可以通过更基础的API来读取模型，指定使用单文件格式:
```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.ckpt_manager import OfficialSD15Format
from rainbowneko.ckpt_manager import LocalCkptSource, NekoLoader

wrapper=SD15Wrapper.from_pretrained(
    _partial_=True,
    models=NekoLoader(
        format=OfficialSD15Format(), # 指定单文件格式
        source=LocalCkptSource(), # 从路径读取 (可以是下载链接)
    ).load(path='ckpt/DreamShaper.safetensors', _partial_=True)
),
```
````

**保存模型**

🚧 开发中

:::


::::

## lora模型格式

::::{tab-set}
:::{tab-item} HCP-Diffusion格式

**读取模型**

HCP-Diffusion将lora作为一种插件，与RainbowNeko Engine中定义的插件格式相同。HCP-Diffusion提供了专用的LoRA加载器`HCPLoraLoader`，可以根据lora权重文件自动推断LoRA结构和参数，自动为模型添加LoRA插件。

比如在workflow中加载LoRA:
```python
from rainbowneko.infer import LoadModelAction
from hcpdiff.ckpt_manager import HCPLoraLoader

LoadModelAction(cfg=dict(
    lora_unet=HCPLoraLoader(
        path='ckpts/lora_unet-1000.safetensors', # LoRA路径
        state_prefix='denoiser.', # 去除模型参数文件中模型路径前缀
        alpha=1, # LoRA的权重
    )
), key_map_in=('denoiser -> model', 'in_preview -> in_preview')),
```

````{note}
LoRA也可以像其他插件一样，先构建，再读取参数:
```python
from rainbowneko.infer import BuildPluginAction, LoadModelAction
from rainbowneko.parser import CfgWDPluginParser
from rainbowneko.ckpt_manager import NekoPluginLoader
from hcpdiff.models.lora_layers_patch import LoraLayer

# build LoRA
BuildPluginAction(parser=CfgWDPluginParser(cfg_plugin=dict(
    lora_unet=LoraLayer.wrap_model(
        _partial_=True,
        lr=1e-4,
        rank=4,
        alpha=2,
        layers=[
            're:.*\.attn.?$',
            're:.*\.ff$',
        ]
    )
)), key_map_in=('denoiser -> model', 'in_preview -> in_preview')),


# load LoRA
LoadModelAction(cfg=dict(
    lora_unet=NekoPluginLoader(
        path='ckpts/lora_unet-1000.safetensors',
        state_prefix='denoiser.',
        alpha=1,
    ),
), key_map_in=('denoiser -> model', 'in_preview -> in_preview')),
```
````

````{attention}
如果要在训练中接着之前的lora训练，请使用`NekoPluginLoader`，不要使用`HCPLoraLoader`。`HCPLoraLoader`会添加新的LoRA模块。
````

**保存模型**

保存LoRA时也同样作为插件储存，比如在训练中保存指定的LoRA插件:
```python
from rainbowneko.ckpt_manager import plugin_saver

ckpt_saver=dict(
    lora_unet=plugin_saver(
        ckpt_type='safetensors',
        target_plugin='lora1', # 保存model_plugin中定义的名称为"lora1"的插件
    )
),
```

:::

:::{tab-item} webui与comfyui格式

**读取模型**

HCP-Diffusion可以读取webui格式的lora模型，自动转换成HCP-Diffusion的插件。

比如读取只有unet的lora模型:
```python
from rainbowneko.ckpt_manager import SafeTensorFormat
from hcpdiff.ckpt_manager import LoraWebuiFormat, HCPLoraLoader

LoadModelAction(cfg=dict(
    lora1=HCPLoraLoader(
        format=LoraWebuiFormat(
            format=SafeTensorFormat()
        ), # 指定webui格式
        path='exps/lora_paimeng/ckpts/lora-webui.safetensors',
        state_prefix='denoiser.',
        alpha=1,
    )
), key_map_in=('denoiser -> model', 'in_preview -> in_preview'))
```

读取同时有unet和text-encoder的lora模型:
```python
from rainbowneko.ckpt_manager import SafeTensorFormat
from hcpdiff.ckpt_manager import LoraWebuiFormat, HCPLoraLoader

# U-Net
LoadModelAction(cfg=dict(
    lora1=HCPLoraLoader(
        format=LoraWebuiFormat(
            format=SafeTensorFormat()
        ), # 指定webui格式
        path='exps/lora_paimeng/ckpts/lora-webui.safetensors',
        state_prefix='denoiser.',
        alpha=1,
    )
), key_map_in=('denoiser -> model', 'in_preview -> in_preview')),

# Text-Encoder
LoadModelAction(cfg=dict(
    lora1=HCPLoraLoader(
        format=LoraWebuiFormat(
            format=SafeTensorFormat()
        ), # 指定webui格式
        path='exps/lora_paimeng/ckpts/lora-webui.safetensors',
        state_prefix='TE.',
        alpha=1,
    )
), key_map_in=('TE -> model', 'in_preview -> in_preview'))
```

**保存模型**

HCP-Diffusion可以将lora保存为webui格式，可以直接在webui或者comfyui中使用。只需要在配置文件中指定储存格式:
```python
from rainbowneko.ckpt_manager import NekoPluginSaver, SafeTensorFormat
from hcpdiff.ckpt_manager import LoraWebuiFormat

ckpt_saver=dict(
    lora_unet=NekoPluginSaver(
        format=LoraWebuiFormat(format=SafeTensorFormat()), # webui格式，储存成safetensors格式
        target_plugin='lora1',
    )
),
```

:::

::::
