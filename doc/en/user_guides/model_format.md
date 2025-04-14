# Model File Format

Currently, both base models and LoRA models come in various formats, with different frameworks supporting different ones. HCP-Diffusion supports reading and saving models in multiple formats through configuration settings. These include the HCP-Diffusion native format, the `diffusers` format, WebUI-compatible `.ckpt` format, and LoRA models in WebUI format, among others.

## Base Model Formats

Base models are available as either a single `.ckpt` or `.safetensors` file (supported by WebUI and ComfyUI), or in the `diffusers` format, where model components are stored in a folder structure. HCP-Diffusion allows you to flexibly read and save any of these formats through its config.

::::{tab-set}

:::{tab-item} `diffusers` Format

**Loading Models**

HCP-Diffusion provides simplified loaders, `SD15_auto_loader` and `SDXL_auto_loader`, that automatically detect whether the provided path is a single file or a `diffusers` folder format, reducing the need for manual setup:

```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.easy import SD15_auto_loader, SDXL_auto_loader

wrapper = SD15Wrapper.from_pretrained(  # Load pretrained base model
    _partial_=True,
    models=SD15_auto_loader(ckpt_path='Lykon/DreamShaper', _partial_=True),
),
```

````{note}
You can also use the lower-level API for more control and explicitly specify the format:

```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.ckpt_manager import DiffusersSD15Format
from rainbowneko.ckpt_manager import LocalCkptSource, NekoLoader

wrapper = SD15Wrapper.from_pretrained(
    _partial_=True,
    models=NekoLoader(
        format=DiffusersSD15Format(),  # Explicitly use diffusers format
        source=LocalCkptSource(),      # Load from local path or HuggingFace repo
    ).load(path='Lykon/DreamShaper', _partial_=True)
),
```
````

**Saving Models**

🚧 Work in progress

:::

:::{tab-item} Single File Format

**Loading Models**

HCP-Diffusion also supports single file formats such as `.ckpt` or `.safetensors`. The `SD15_auto_loader` and `SDXL_auto_loader` simplify loading by automatically identifying the format:

```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.easy import SD15_auto_loader, SDXL_auto_loader

wrapper = SD15Wrapper.from_pretrained(
    _partial_=True,
    models=SD15_auto_loader(ckpt_path='ckpt/DreamShaper.safetensors', _partial_=True),
),
```

````{note}
You can also load the model using the lower-level API with format explicitly set:

```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.ckpt_manager import OfficialSD15Format
from rainbowneko.ckpt_manager import LocalCkptSource, NekoLoader

wrapper = SD15Wrapper.from_pretrained(
    _partial_=True,
    models=NekoLoader(
        format=OfficialSD15Format(),  # Use single-file format
        source=LocalCkptSource(),     # Load from local path or URL
    ).load(path='ckpt/DreamShaper.safetensors', _partial_=True)
),
```
````

**Saving Models**

🚧 Work in progress

:::

::::

## LoRA Model Formats

::::{tab-set}

:::{tab-item} HCP-Diffusion Format

**Loading Models**

LoRA modules are treated as plugins in HCP-Diffusion, consistent with the plugin system used by the RainbowNeko Engine. The built-in `HCPLoraLoader` can automatically infer LoRA structure and parameters from the weight file and integrate them into the model.

Example usage in a workflow:

```python
from rainbowneko.infer import LoadModelAction
from hcpdiff.ckpt_manager import HCPLoraLoader

LoadModelAction(cfg=dict(
    lora_unet=HCPLoraLoader(
        path='ckpts/lora_unet-1000.safetensors',  # Path to LoRA file
        state_prefix='denoiser.',                # Strip prefix from weight keys
        alpha=1,                                  # LoRA scaling factor
    )
), key_map_in=('denoiser -> model', 'in_preview -> in_preview')),
```

````{note}
You can also manually build the LoRA plugin first and then load its parameters:

```python
from rainbowneko.infer import BuildPluginAction, LoadModelAction
from rainbowneko.parser import CfgWDPluginParser
from rainbowneko.ckpt_manager import NekoPluginLoader
from hcpdiff.models.lora_layers_patch import LoraLayer

# Build LoRA
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

# Load LoRA
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
To resume training from a previous LoRA checkpoint, use `NekoPluginLoader` instead of `HCPLoraLoader`. The latter will add a *new* LoRA module.
````

**Saving Models**

LoRA plugins can also be saved similarly. For example, during training, you can save a specific LoRA plugin:

```python
from rainbowneko.ckpt_manager import plugin_saver

ckpt_saver = dict(
    lora_unet=plugin_saver(
        ckpt_type='safetensors',
        target_plugin='lora1',  # Save the plugin named "lora1" in model_plugin
    )
),
```

:::

:::{tab-item} WebUI and ComfyUI Format

**Loading Models**

HCP-Diffusion supports loading LoRA models in the WebUI format and converting them into its own plugin system.

Example of loading a LoRA model with only a U-Net:

```python
from rainbowneko.ckpt_manager import SafeTensorFormat
from hcpdiff.ckpt_manager import LoraWebuiFormat, HCPLoraLoader

LoadModelAction(cfg=dict(
    lora1=HCPLoraLoader(
        format=LoraWebuiFormat(
            format=SafeTensorFormat()
        ),
        path='exps/lora_paimeng/ckpts/lora-webui.safetensors',
        state_prefix='denoiser.',
        alpha=1,
    )
), key_map_in=('denoiser -> model', 'in_preview -> in_preview'))
```

Example with both U-Net and Text Encoder:

```python
# U-Net
LoadModelAction(cfg=dict(
    lora1=HCPLoraLoader(
        format=LoraWebuiFormat(format=SafeTensorFormat()),
        path='exps/lora_paimeng/ckpts/lora-webui.safetensors',
        state_prefix='denoiser.',
        alpha=1,
    )
), key_map_in=('denoiser -> model', 'in_preview -> in_preview')),

# Text Encoder
LoadModelAction(cfg=dict(
    lora1=HCPLoraLoader(
        format=LoraWebuiFormat(format=SafeTensorFormat()),
        path='exps/lora_paimeng/ckpts/lora-webui.safetensors',
        state_prefix='TE.',
        alpha=1,
    )
), key_map_in=('TE -> model', 'in_preview -> in_preview'))
```

**Saving Models**

HCP-Diffusion supports saving LoRA modules in WebUI-compatible format, allowing direct use in WebUI or ComfyUI. Just specify the format in your config:

```python
from rainbowneko.ckpt_manager import NekoPluginSaver, SafeTensorFormat
from hcpdiff.ckpt_manager import LoraWebuiFormat

ckpt_saver = dict(
    lora_unet=NekoPluginSaver(
        format=LoraWebuiFormat(format=SafeTensorFormat()),  # Save as WebUI format in .safetensors
        target_plugin='lora1',
    )
),
```

:::

::::
