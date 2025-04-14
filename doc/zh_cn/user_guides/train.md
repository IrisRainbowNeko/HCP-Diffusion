from socket import fromfd

# 模型训练

HCP-Diffusion可以通过```python```配置文件，配置各种训练阶段可能会用到的组件。
包括模型结构，训练参数和方式，数据集配置等。

## 训练基础配置
与训练相关的基础配置文件与示例在```cfgs/train```目录中。所有的训练配置文件都应
同时继承```train_base.py```与```tuning_base.py```.
```train_base.py```中定义了训练阶段需要使用的各种超参数，以及数据集的相关配置。
```tuning_base.py```中则定义了训练阶段的模型结构和训练参数。哪些模型参数和插件需要被训练，哪些层需要以怎样的方式添加lora。

::::{tab-set}
:::{tab-item} 单GPU训练

```bash
hcp_train_1gpu --cfg cfgs/train/py/配置文件.py
```

配置文件中的值，可以在cli中修改:
```bash
hcp_train_1gpu --cfg cfgs/train/py/配置文件.py model.wrapper.models.ckpt_path=pretrained_model_path data_train.dataset1.batch_size=8
```
:::

:::{tab-item} 多GPU训练
多GPU训练需要在`cfgs/launcher/multi.yaml`中指定训练用到的GPU id和GPU数量，然后运行:
```bash
hcp_train --cfg cfgs/train/py/配置文件.py
```

配置文件中的值，可以在cli中修改:
```bash
hcp_train --cfg cfgs/train/py/配置文件.py model.wrapper.models.ckpt_path=pretrained_model_path data_train.dataset1.batch_size=8
```
:::
::::

## 模型配置

### 基础模型配置
基础模型的配置在`model`中进行配置，`model.wrapper`中定义了模型的结构，初始化模型的相关配置。比如加载预训练的SD1.5模型:

```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.easy import SD15_auto_loader

wrapper=SD15Wrapper.from_pretrained( # 模型封装
    _partial_=True,
    models=SD15_auto_loader(ckpt_path='Lykon/DreamShaper', _partial_=True), # 简化的预训练模型加载器
),
```

```{note}
也可以通过更基础的API来读取模型，详细配置见 [模型文件格式说明](./model_format.md)
```

### 替换VAE

::::{tab-set}
:::{tab-item} diffusers格式VAE

在wrapper配置中，可以单独指定VAE模块，使用其他VAE模型:
```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.easy import SD15_auto_loader
from diffusers import AutoencoderKL

wrapper=SD15Wrapper.from_pretrained( # 模型封装
    _partial_=True,
    models=SD15_auto_loader(
        ckpt_path='Lykon/DreamShaper',
        vae=AutoencoderKL.from_pretrained('vae/'),
        _partial_=True
    ), # 简化的预训练模型加载器
),
```

:::

:::{tab-item} 单文件格式 (webui格式)

在wrapper配置中，可以单独指定VAE模块，使用其他VAE模型:
```python
from hcpdiff.models import SD15Wrapper
from hcpdiff.easy import SD15_auto_loader
from diffusers import AutoencoderKL

wrapper=SD15Wrapper.from_pretrained( # 模型封装
    _partial_=True,
    models=SD15_auto_loader(
        ckpt_path='Lykon/DreamShaper',
        vae=AutoencoderKL.from_single_file('vae.ckpt'),
        _partial_=True
    ), # 简化的预训练模型加载器
),
```

:::
::::

## 数据集配置
HCP-Diffusion可以定义多个并行数据集，每一步训练，会从所有数据集中各抽取一个batch进行前向和反向传播，并把他们的梯度加起来。每个数据集都是独立运算的，所以图片尺寸，数据格式都可以不一样。配置每个数据集的batch size和loss_weight可以调节他们的比例与权重。

示例:
```python
from hcpdiff.data import TextImagePairDataset

data_train=dict(
    dataset1=TextImagePairDataset(_partial_=True, batch_size=4, loss_weight=1.0,
        ...
    ),
    dataset2=TextImagePairDataset(_partial_=True, batch_size=1, loss_weight=1.0,
        ...
    ),
)
```

```{tip}
如果两个数据集`dataset1`和`dataset2`的重要程度是a:b，那么他们应该满足: $\frac{batch\_size_1 \times loss\_weight_1}{batch\_size_2 \times loss\_weight_2} = \frac{a}{b}$
```

更详细的介绍见 [RainbowNeko Engine数据配置](https://rainbownekoengine.readthedocs.io/zh-cn/stable/train/data.html)

### 数据源

每个数据集都可以定义多个数据源。一个数据集中的所有数据源，都会一起被分bucket，打乱，处理，相当于合并这几组数据。

示例:
```python
from hcpdiff.data import TextImagePairDataset, Text2ImageSource

dataset1=TextImagePairDataset(_partial_=True, batch_size=4, loss_weight=1.0,
    source=dict(
        data_source1=Text2ImageSource(
            img_root= 'imgs1/',
            label_file= '${.img_root}',  # 标签文件路径，与图片相同
            prompt_template='prompt_template/caption.txt', # prompt模板
            repeat=1, # 数据源可以通过复制N次来调节比例
        ),
        data_source2=Text2ImageSource(
            img_root= 'imgs2/',
            label_file= '${.img_root}',  # 标签文件路径，与图片相同
            prompt_template='prompt_template/caption.txt',
        ),
    ),
    ...
)
```

更详细的介绍见 [RainbowNeko Engine配置](https://rainbownekoengine.readthedocs.io/zh-cn/stable/train/data.html#data-source-configuration)


### Bucket
Bucket可以将图像分组排列组合，把具有相同特性的图像放入同一个batch中。

支持的Bucket如下:
+ FixedBucket: 将所有图像缩放并裁剪到给定的相同尺寸。
```python
from rainbowneko.data import FixedBucket
FixedBucket(
    target_size=[512,512] # 使用512x512的固定分辨率
)
```
+ RatioBucket (ARB): 将图像按宽高比分组，各个batch可以有不同宽高比。减少图像裁剪带来的损耗。
    + from ratios: 根据给定的宽高比范围，自动筛选出和目标尺寸最接近的n个不同宽高比的bucket。
    ```python
    from rainbowneko.data import RatioBucket
    RatioBucket.from_ratios(
        target_area=512*512, # 等比缩放图片使分辨率尽可能接近512x512
        num_bucket=6, # 桶的数量
        
        # 可选参数
        step_size=8, # 桶的步长
        ratio_max=4, # 最大宽高比
        pre_build_bucket='path.pkl', # 桶的储存路径，构建一次可以存起来
    )
    ```
    + from_files: 根据训练用到的图像自动对宽高比进行聚类，选出与目标尺寸最接近的n个bucket。
    ```python
    from rainbowneko.data import RatioBucket
    RatioBucket.from_files(
        target_area=512*512, # 等比缩放图片使分辨率尽可能接近512x512
        num_bucket=6, # 桶的数量
        
        # 可选参数
        step_size=8, # 桶的步长
        pre_build_bucket='path.pkl', # 桶的储存路径，构建一次可以存起来
    )
    ```
+ SizeBucket: 将图像按分辨率分组，各个batch可以有不同分辨率。减少图像裁剪和缩放带来的损耗。
    + from_files: 根据训练用到的图像自动对分辨率进行聚类，选出与目标尺寸最接近的n个bucket。
    ```python
    from rainbowneko.data import SizeBucket
    SizeBucket.from_files(
        num_bucket=6, # 桶的数量
        
        # 可选参数
        step_size=8, # 桶的步长
        pre_build_bucket='path.pkl', # 桶的储存路径，构建一次可以存起来
    )
    ```
+ LongEdgeBucket: 将图像长边缩放到固定值后按分辨率分组，各个batch可以有不同分辨率。减少图像裁剪带来的损耗。
    + from_files: 根据训练用到的图像自动对分辨率进行聚类，选出与目标尺寸最接近的n个bucket。
    ```python
    from rainbowneko.data import LongEdgeBucket
    LongEdgeBucket.from_files(
        target_edge=800, # 将长边缩放到800
        num_bucket=6, # 桶的数量
        
        # 可选参数
        step_size=8, # 桶的步长
        pre_build_bucket='path.pkl', # 桶的储存路径，构建一次可以存起来
    )
    ```

### 高级数据集配置

:::{dropdown} 添加正则化数据集
:animate: fade-in

正则化数据集可以用于DreamBooth，或是让模型学习一些自己生成的图像，保留原有生成能力。

准备一个prompt数据集，使用workflow生成图像，作为正则化数据集:
```bash
hcp_run --cfg cfgs/workflow/text2img_dataset.py
```
生成使用的模型和prompt数据集需要在配置文件中进行配置。

```{note}
prompt数据集的格式与常规数据集的格式相同，只是没有图片。
```

:::

:::{dropdown} prompt模板使用
:animate: fade-in

prompt模板可以在训练阶段将其中的占位符替换成指定的文本。
例如一个prompt模板: 

```a photo of a {pt1} on the {pt2}, {caption}```

其中的```{pt1}```和```{pt2}```会被```handler```中被```TemplateFillHandler```替换为指定的词，
这个词可以是自定义的embedding(可以占多个词的位置)，也可以是模型原有的词。

使用示例:
```python
from hcpdiff.data.handler import TemplateFillHandler

TemplateFillHandler(
    word_names={
        'pt1': 'my-cat',
        'pt2': 'sofa',
    },
)
```

````{important}
推荐使用Diffusion常用的简化封装配置:
```python
from hcpdiff.data import StableDiffusionHandler
handler=StableDiffusionHandler(
    bucket=RatioBucket, 
    word_names={
        'pt1': 'my-cat',
        'pt2': 'sofa',
    },
    erase=0,
),
```
````

在训练阶段会将`{pt1}`替换为`my-cat`对应的embedding，将`{pt2}`替换为`sofa`.
`{caption}`则会被替换为图像对应的描述，如果没有定义该图像的描述，则这部分为空。

:::

## Fine-tuning训练配置
当前Fine-tuning支持训练各个组件，通常训练unet和text-encoder。可以单独训练模型的一部分，也可以为各个层分配不同的学习率。
格式如下:
```python
from rainbowneko.parser import CfgWDModelParser

model_part=CfgWDModelParser([
        dict( # 部分1
            lr=1e-5, # 当前部分的学习率
            # weight_decay=1e-2, # 当前部分的weight_decay
            layers=['denoiser'],  # 训练 U-Net
            # layers=['TE'],  # 训练 TextEncoder
        )
    ], 
    weight_decay=1e-2 # 默认weight_decay
),
```

```{note}
描述训练哪些层所用名称是 pytorch 中的模型模块路径，与`model.named_modules()`中的命名一致。可以使用正则表达式批量匹配，比如U-Net的所有self-attention层 `re:denoiser\..*\.attn1$`。

模型配置使用RainbowNeko Engine的配置方式，详细使用方法见 [RainbowNeko Engine配置](https://rainbownekoengine.readthedocs.io/zh-cn/stable/train/first.html)
```

:::{dropdown} 不同学习率同时训练多个部分
:animate: fade-in

同时训练U-Net和text encoder，使用相同学习率:
```python
from rainbowneko.parser import CfgWDModelParser

model_part=CfgWDModelParser([
        dict(
            lr=1e-5,
            layers=[
                'denoiser',
                'TE',
            ],  # 训练 U-Net 和 text encoder
        ),
    ], 
    weight_decay=1e-2 # 默认weight_decay
),
```

使用不同学习率:
```python
from rainbowneko.parser import CfgWDModelParser

model_part=CfgWDModelParser([
        dict( # 部分1
            lr=1e-5, # 当前部分的学习率
            layers=['denoiser'],  # 训练 U-Net
        ),
        dict( # 部分2
            lr=2e-6, # 当前部分的学习率
            layers=['TE'],  # 训练 text encoder
        )
    ], 
    weight_decay=1e-2 # 默认weight_decay
),
```

:::

## Prompt-tuning训练配置
prompt-tuning训练word embedding，一个word embedding可以占多个词的位置.

首先需要创建自定义word：
```bash
python -m hcpdiff.tools.create_embedding 预训练模型路径 word名称 一个word占几个词 [--init_text 初始化单词]
# 随机初始化 --init_text *[标准差, word长度]
# 部分随机 --init_text cat, *[标准差, word长度], tail
```

添加配置`emb_pt`指定需要训练的word：
```python
from hcpdiff.parser import CfgEmbPTParser
emb_pt=CfgEmbPTParser(
    emb_dir='embs/',
    cfg_pt={
        'pt-paimeng': dict(lr=0.003, weight_decay=1e-2)
    }
),
```

## LoRA训练配置
当前LoRA支持在模型中任意```Linear```和```Conv2d```层中添加。
配置方式与Fine-tuning类似，不过LoRA作为插件添加:
```python
from rainbowneko.parser import CfgWDPluginParser
from hcpdiff.models.lora_layers_patch import LoraLayer
model_plugin=CfgWDPluginParser(cfg_plugin=dict(
    lora1=LoraLayer.wrap_model(
        _partial_=True, # 必须加
        lr=1e-4, # 这个插件的学习率
        rank=4, # LoRA的维度 (LoRA这个插件的参数)
        alpha=2, # LoRA的权重，一般与rank相同或为rank的一半
        layers=[
            're:denoiser.*\.attn.?$', # Attention层
            're:denoiser.*\.ff$', # FeedForward层
        ]
    )
), weight_decay=0.1), # 所有插件的默认weight_decay
```

## 高级配置

:::{dropdown} loss weight mask (为图上每个区域设置不同重要性)
:animate: fade-in


![](../../imgs/att_map.webp)

在训练图像较少的情况下，模型难以归纳出什么是重要的特征。所以可以通过添加loss mask，让模型训练阶段更多或更少关注一部分特征。
如上图所示。

loss mask和原始图像应放置于不同文件夹中，并且有着相同的文件名。

loss mask是一个灰度图，其亮度值与注意力倍率如下图所示。

| 亮度  | 0% | 25% | 50%  | 75%  | 100% |
|-----|----|-----|------|------|------|
| 倍率  | 0% | 50% | 100% | 300% | 500% |


:::

### CLIP skip
有些模型在训练阶段会跳过几个CLIP的block，在`model.wrapper`中的`TE_hook_cfg`参数中可以设置跳过CLIP几个block，默认为0(与webui中的clip skip=1等价)，不跳过任何层。

````{tip}
比如跳过一层:
```python
TE_hook_cfg=TEHookCFG(clip_skip=1)
```
````