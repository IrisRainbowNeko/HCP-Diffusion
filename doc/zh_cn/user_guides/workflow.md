# 图像生成

HCP-Diffusion基于workflow来生成图像，workflow的配置文件同样是一个`python`文件，可以通过`.py`文件来描述生成图像的过程。使用`workflow`可以在生成过程中加入超分，局部修改等各种操作。甚至可以让每个`step`都使用不同的`prompt`，CFG强度，或使用不同的模型。

```bash
# 运行workflow
hcp_run --cfg cfgs/workflow/text2img.yaml
```

## 单词注意力加强
```{note}
生成图像时可以单独加强部分词的注意力:

格式为: {加强的文本:倍率}，默认1.1倍
例如：
a {cat} running {in the {city}:1.2}

其中```cat```会被加强1.1倍，```in the```会被加强1.2倍，```city```则会被加强1.2*1.1倍。
```

## 配置基础结构
`workflow`的执行入口在`make_cfg`函数中，返回的`dict`中的`workflow`项是定义的workflow。

### 使用Stable Diffusion生成图像:

:::::{tab-set}
::::{tab-item} 简化配置 (适合小白)

`cfgs/workflow/easy/text2img.yaml`提供了简化的图像生成配置，只需要设置少量参数选项就可以生成图像，但是自由度较低，可以实现的功能有限。

常用配置:
```python
from hcpdiff.easy.cfg import SD15_t2i
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SD15_t2i(
        pretrained_model='Lykon/DreamShaper', # 预训练模型路径
        prompt='masterpiece, best quality, 1girl, cat ears, outside', # 正面描述
        # negative_prompt='', # 添加负面描述
        bs=4, # 批次大小
        width=512, # 图片宽度
        height=512, # 图片高度
        guidance_scale=7.0 # CFG引导强度
    )
```

如果需要修改采样器，可以配置参数`noise_sampler`，默认为`dpmpp_2m_karras`:
```python
from hcpdiff.easy import Diffusers_SD

SD15_t2i(
        pretrained_model='Lykon/DreamShaper', # 预训练模型路径
        prompt='masterpiece, best quality, 1girl, cat ears, outside', # 正面描述
        bs=4, # 批次大小
        width=512, # 图片宽度
        height=512, # 图片高度
        guidance_scale=7.0, # CFG引导强度
        noise_sampler=Diffusers_SD.euler_a # 替换采样器
    )
```

可以选择的采样器:

| 采样器          | 说明                     |
|-----------------|--------------------------|
| dpmpp_2m        | 步数少                   |
| dpmpp_2m_karras | 步数少，效果好，比较常用 |
| ddim            | 步数较多                 |
| euler           |                          |
| euler_a         | 二次元常用，比较平滑     |

其他配置:
```python
from hcpdiff.easy.cfg import SD15_t2i
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SD15_t2i(
        pretrained_model='Lykon/DreamShaper', # 预训练模型路径
        prompt='masterpiece, best quality, 1girl, cat ears, outside', # 正面描述
        # negative_prompt='', # 添加负面描述
        bs=4, # 批次大小
        width=512, # 图片宽度
        height=512, # 图片高度
        guidance_scale=7.0, # CFG引导强度
        
        seed=42, # 固定随机种子
        N_steps=30, # 设置采样步数
        save_root='output_pipe/', # 输出结果保存目录
    )
```

::::

::::{tab-item} 完整配置 (适合了解模型结构的)

`cfgs/workflow/text2img.yaml`提供了图像生成工作流配置，自由度高，可以实现各种功能。文生图功能一般由以下几个模块构成。其他更多的工作流可以参考`cfgs/workflow/`目录下的配置文件。

:::{dropdown} 模型加载
:animate: fade-in

```python
import torch
from rainbowneko.parser import neko_cfg
from hcpdiff.easy import Diffusers_SD, SD15_auto_loader
from rainbowneko.infer import Actions, PrepareAction
from hcpdiff.workflow import BuildModelsAction

@neko_cfg
def build_model(pretrained_model='ckpts/any5', noise_sampler=Diffusers_SD.dpmpp_2m_karras) -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=torch.float16), # 配置设备和精度
        BuildModelsAction( # 构建模型，此次加载预训练的
            model_loader=SD15_auto_loader(_partial_=True,
                ckpt_path=pretrained_model,
                noise_sampler=noise_sampler # 设置采样器，预设
            )
        ),
    ])
```

````{note}
这里的noise_sampler使用了预设的简化配置，预设的配置包括:

| 采样器          | 说明                     |
|-----------------|--------------------------|
| dpmpp_2m        | 步数少                   |
| dpmpp_2m_karras | 步数少，效果好，比较常用 |
| ddim            | 步数较多                 |
| euler           |                          |
| euler_a         | 二次元常用，比较平滑     |

如果需要其他采样器配置，可以使用完整的配置:
```python
from diffusers import DPMSolverMultistepScheduler
from hcpdiff.diffusion.sampler import DiffusersSampler

# 使用diffusers的采样器
noise_sampler=DiffusersSampler(
    DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        algorithm_type='sde-dpmsolver++',
        use_karras_sigmas=True,
    )
)
```
````

:::

:::{dropdown} 模型优化
:animate: fade-in

```python
from hcpdiff.workflow import PrepareDiffusionAction, XformersEnableAction, VaeOptimizeAction

@neko_cfg
def optimize_model() -> Actions:
    return Actions([
        PrepareDiffusionAction(), # 配置模型相关参数
        XformersEnableAction(), # 开启xformers优化显存
        VaeOptimizeAction(slicing=True), # VAE相关优化，减少显存
    ])
```

:::

:::{dropdown} 文本编码
:animate: fade-in

```python
from hcpdiff.workflow import TextHookAction, AttnMultTextEncodeAction

@neko_cfg
def text(prompt=prompt, negative_prompt=negative_prompt, bs=4, N_repeats=1, layer_skip=1) -> Actions:
    return Actions([
        # 支持高级文本编码器功能。
        # N_repeats: 最大prompt长度拓展
        # layer_skip: 跳过最后N层 (clip skip), 这里0和webui的1一样
        TextHookAction(N_repeats=N_repeats, layer_skip=layer_skip),
        # 文本编码操作，支持token权重
        AttnMultTextEncodeAction(
            prompt=prompt, # 正面提示词
            negative_prompt=negative_prompt, # 负面提示词
            bs=bs # 批次大小
        ),
    ])
```

:::

:::{dropdown} 图像信息设置
:animate: fade-in

```python
from hcpdiff.workflow import SeedAction, MakeTimestepsAction, MakeLatentAction

@neko_cfg
def config_diffusion(seed=42, N_steps=20, width=512, height=512) -> Actions:
    return Actions([
        SeedAction(seed), # 设置随机种子
        MakeTimestepsAction(N_steps=N_steps), # 设置采样步数
        # MakeTimestepsAction(N_steps=N_steps, strength=0.6), # 去噪强度，用于图生图
        MakeLatentAction(width=width, height=height) # 设置图像大小
    ])
```

:::

:::{dropdown} 图像生成
:animate: fade-in

```python
from rainbowneko.infer import LoopAction
from hcpdiff.workflow import DiffusionStepAction, time_iter

@neko_cfg
def diffusion(guidance_scale=7.0) -> Actions:
    return Actions([
        LoopAction( # 循环执行Action
            iterator=time_iter, # 时间步迭代器 [{'t':t} for t in timesteps]
            actions=[
                DiffusionStepAction(guidance_scale=guidance_scale) # 进行一步去噪
            ]
        )
    ])
```

:::

:::{dropdown} 图像解码
:animate: fade-in

```python
from hcpdiff.workflow import DecodeAction, SaveImageAction

@neko_cfg
def decode() -> Actions:
    return Actions([
        DecodeAction(), # 使用VAE解码latent变成图像
        SaveImageAction(save_root='output_pipe/', image_type='png'), # 保存图像
    ])
```

:::

::::
:::::

## 支持的Action

TODO