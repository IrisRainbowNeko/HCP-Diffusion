# lora训练指南

这个指南将会教你如何训练一个Lora模型，为SD底模添加一些新的知识。

这个教程过程总共分为以下几个部分：
* 准备数据集
* 训练模型
* 使用lora生成图像
* 保存为sd-webui格式

## 准备数据集

准备你要训练的数据集，数据集应该包含一些文本标注和对应的图像，文本和图像应该是一一对应的关系。
将要训练的图片放在一个文件夹中，比如放在`train_data`文件夹中，结构如下：
```text
train_data
├── 1.png
├── 2.png
├── 3.png
├── ......
```


文本标注支持多种格式，框架可以自动识别。更详细的说明见 [RainbowNeko Engine](https://rainbownekoengine.readthedocs.io/zh-cn/stable/train/label_file.html)

::::{tab-set}
:::{tab-item} txt格式

`txt`格式的标注，与对应的图片文件名相同，只是后缀不同，比如`1.txt`，`2.txt`等。
`txt`内写图片对应的标注。
可以放在和图片同一个文件夹中。
```text
train_data
├── 1.png
├── 1.txt
├── 2.png
├── 2.txt
├── 3.png
├── 3.txt
├── ......
```

:::

:::{tab-item} json格式

`json`格式的标注，所有图片的标注放在一个json文件中，格式如下：
```json
{
    "1": "标注1",
    "2": "标注2",
    "3": "标注3",
    ......
}
```

:::

:::{tab-item} yaml格式

`yaml`格式的标注，所有图片的标注放在一个yaml文件中，格式如下：
```yaml
"1": "标注1"
"2": "标注2"
"3": "标注3"
......
```

:::

::::

## 模型训练

准备好数据集后，我们可以开始训练模型了。
推荐安装`tensorboard`或`wandb`以更清晰的查看训练进度。
```bash
# 安装tensorboard
pip install tensorboard
# 安装wandb
pip install wandb
```

### 训练配置文件填写

::::{tab-set}
:::{tab-item} 常规配置文件

在`cfgs/train/py/examples`中有提供训练相关的模板文件，其中`lora_preview.py`提供了lora模型训练的模板，带有预览功能，我们可以继承这个模板文件来进行训练。

模板默认只为`U-Net`部分添加lora，不在`text encoder`部分添加lora。

`lora_preview.py`继承自`SD_FT.py`，其中主要需要修改的配置如下:
```python
...
model_plugin=CfgWDPluginParser(cfg_plugin=dict(
    lora1=LoraLayer.wrap_model(
        _partial_=True,
        lr=1e-4, # lora学习率
        rank=4, # lora的维度
        alpha=2, # lora的权重，一般与rank相同或为rank的一半
        layers=[
            're:denoiser.*\.attn.?$',
            're:denoiser.*\.ff$',
        ]
    )
), weight_decay=0.1), # weight_decay一般为0.1或者0.01
...
train=dict(
    train_steps=1000, # 训练总步数
    # train_epochs=10, # 可以使用epochs来代替步数，指定训练轮数
    save_step=200, # 保存模型步数间隔
    
    # 优化器，如果显存不够可以换成8bit优化器
    optimizer=torch.optim.AdamW(_partial_=True, betas=(0.9, 0.99)),
    
    # 学习率曲线，默认恒定不变
    scheduler=ConstantLR(
        _partial_=True,
        warmup_steps=0,
    ),
),
...
model=dict(
    name='model',
    
    # ckpt_path是预训练底模的路径，改成需要的
    wrapper=SD15Wrapper.from_pretrained(
        models=SD15_auto_loader(ckpt_path='Lykon/DreamShaper', _partial_=True),
        _partial_=True,
    ),
),
...
evaluator=HCPPreviewer(_partial_=True,
    interval=100, # 预览间隔
    workflow=t2i_lora, # 使用哪个工作流预览图像
),
```
在预览图像时，底模和lora等插件都不会加载新的，会使用训练中使用的模型。

对于数据集配置:
```python
@neko_cfg
def cfg_data():
    return dict(
        dataset1=TextImagePairDataset(_partial_=True,
            batch_size=4, # 批次大小
            loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root= 'imgs/', # 图片路径
                    label_file= '${.img_root}',  # 标注路径，默认与图片一致
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(
                bucket=RatioBucket, # 使用ARB
                word_names=dict(pt1='paimeng'), # 触发词是paimeng
                erase=0,
            ),
            bucket=RatioBucket.from_files(
                target_area=512*512, # 图片训练分辨率
                num_bucket=4, # 分桶数量，训练集分辨率越多，分桶数量越多
            ),
            cache=VaeCache(bs=4) # 缓存VAE编码，减少显存使用
        )
    )
```

如果需要把lora同时保存成HCP-Diffusion的格式和webui的格式，可以在`ckpt_saver`中配置:
```python
from rainbowneko.ckpt_manager import NekoPluginSaver, SafeTensorFormat
from hcpdiff.ckpt_manager import LoraWebuiFormat

ckpt_saver=dict(
    _replace_ = True,
    lora_unet=NekoPluginSaver(
        format=SafeTensorFormat(),
        target_plugin='lora1',
    ),
    lora_unet_webui=NekoPluginSaver(
        format=LoraWebuiFormat(), # webui格式
        target_plugin='lora1',
    ),
),
```

:::

:::{tab-item} 简化配置文件

在`cfgs/train/py/easy`中有提供训练相关的简化模板文件，其中`SD15_lora_preview.py`提供了lora模型训练的模板，带有预览功能，我们可以基于这个模板文件来进行训练。

模板默认只为`U-Net`部分添加lora，不在`text encoder`部分添加lora。

```python
from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SD15_lora_train, cfg_data_SD_ARB, SD15_t2i
from hcpdiff.evaluate import HCPPreviewer

# 预览使用的prompt
prompt = ('paimeng, 1girl, halo, white_hair, solo, smile, blue_eyes, looking_at_viewer, open_mouth, long_sleeves, white_dress, dress, single_thighhigh,'
          ' :d, cape, hair_between_eyes, thighhighs, hair_ornament, blush, white_outline, outline, sky, scarf, cloud, white_thighhighs, arm_up,'
          ' notice_lines, paimon_(genshin_impact)')

@neko_cfg
def make_cfg():
    return dict(
        **SD15_lora_train(
            base_model='Lykon/DreamShaper', # 预训练模型路径
            train_steps=1000, # 训练总步数
            save_step=200, # 保存模型步数间隔
            rank=8, # lora的维度            
            dataset=dict(
                dataset1=cfg_data_SD_ARB( # 使用ARB使图像尽可能不被剪裁
                    img_root='imgs/', # 训练图片路径，标注和图片在同一个文件夹
                    batch_size=4, # 批次大小
                    trigger_word='paimeng', # 触发词
                    resolution=512*512, # 训练分辨率
                    num_bucket=4, # 分桶数量，训练集分辨率越多，分桶数量越多
                )
            )
        ),
        # 开启预览
        evaluator=HCPPreviewer(_partial_=True,
            interval=100, # 预览间隔
            workflow=SD15_t2i(
                pretrained_model='${model.wrapper.models.ckpt_path}',
                prompt=prompt, # 预览使用的prompt
                seed=42, # 固定随机种子
            ),
        ),
    )
```

在这个配置基础上，还有一些其他配置可以选择:
```python
@neko_cfg
def make_cfg():
    return dict(
        **SD15_lora_train(
            base_model='Lykon/DreamShaper', # 预训练模型路径
            train_steps=1000, # 训练总步数
            save_step=200, # 保存模型步数间隔
            rank=8, # lora的维度
            
            lr=1e-4, # lora学习率
            alpha=8, # lora的权重，一般与rank相同或为rank的一半
            clip_skip=0, # clip跳过的层数，0表示不跳过
            with_conv=False, # 是否在卷积层添加lora
            low_vram=False, # 使用8bit优化器
            save_webui_format=True, # 储存成webui的格式
            
            dataset=dict(
                dataset1=cfg_data_SD_ARB( # 使用ARB使图像尽可能不被剪裁
                    img_root='imgs/', # 训练图片路径，标注和图片在同一个文件夹
                    batch_size=4, # 批次大小
                    trigger_word='paimeng', # 触发词
                    resolution=512*512, # 训练分辨率
                    num_bucket=4, # 分桶数量，训练集分辨率越多，分桶数量越多
                )
            )
        ),
        # 开启预览
        evaluator=HCPPreviewer(_partial_=True,
            interval=100, # 预览间隔
            workflow=SD15_t2i(
                pretrained_model='${model.wrapper.models.ckpt_path}',
                prompt=prompt, # 预览使用的prompt
                seed=42, # 固定随机种子
                
                negative_prompt='', # 负面描述
                noise_sampler=Diffusers_SD.dpmpp_2m_karras, # 采样器
                bs=4, # 批次大小
                width=512, # 图片宽度
                height=512, # 图片高度
                N_steps=20, # 采样步数
                guidance_scale=7.0, # CFG引导强度
            ),
        ),
    )
```

:::
::::

### 训练

::::{tab-set}
:::{tab-item} 单GPU训练

```bash
hcp_train_1gpu --cfg cfgs/train/py/examples/lora_preview.py
```

配置文件中的值，可以在cli中修改:
```bash
hcp_train_1gpu --cfg cfgs/train/py/examples/lora_preview.py train.train_epochs=10 # 修改训练轮数
```
:::

:::{tab-item} 多GPU训练
多GPU训练需要在`cfgs/launcher/multi.yaml`中指定训练用到的GPU id和GPU数量，然后运行:
```bash
hcp_train --cfg cfgs/train/py/examples/lora_preview.py
```

配置文件中的值，可以在cli中修改:
```bash
hcp_train --cfg cfgs/train/py/examples/lora_preview.py train.train_epochs=10 # 修改训练轮数
```
:::
::::

训练完毕后，输出的所有结果都会存到一个文件夹中:
```text
exps/2023-07-26-01-05-35
├── cfg.py # 配置文件
├── ckpts # lora模型文件
│   ├── lora_unet-100.safetensors
│   ├── lora_unet-200.safetensors
│   ├── ...
├── tblog # tensorboard日志
│   └── events.out.tfevents.1690346085.myenvironment.210494.0
└── train.log
```

### 进阶配置

自定义lora参数和添加lora的层:
```python
lora_unet=LoraLayer.wrap_model(
    _partial_=True,
    lr=1e-4, # lora学习率
    rank=4, # lora的维度
    alpha=2, # lora的权重，一般与rank相同或为rank的一半
    layers=[
        # k,v层和ff层 (使用正则表达式指定层的名称)
        're:.*\.to_k$'
        're:.*\.to_v$'
        're:.*\.ff$'
    ]
),
lora_TE=LoraLayer.wrap_model(
    _partial_=True,
    lr=2e-5, # lora学习率
    rank=2, # lora的维度
    alpha=2, # lora的权重，一般与rank相同或为rank的一半
    layers=[
        're:.*self_attn$' # 注意力层
        're:.*mlp$' # mlp层
    ]
)
```

## 使用lora生成图像

训练完毕后，我们可以使用lora生成图像。详细的生成配置文件编写请参考[生成配置文件编写](../user_guides/workflow.md)。

这里可以使用简化的配置文件`cfgs/workflow/easy/t2i_lora.py`:
```python
from hcpdiff.easy.cfg import SD15_t2i_lora
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SD15_t2i_lora(
        pretrained_model='Lykon/DreamShaper', # 预训练底模
        lora_info=[
            ('exps/2023-07-26-01-05-35/ckpts/lora_unet-1000.safetensors', 1.0), # (lora文件路径, weight)
        ],
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        bs=4,
        width=512,
        height=512,
        guidance_scale=7.0
    )
```

运行配置文件生成图片:
```shell
hcp_run --cfg cfgs/workflow/easy/t2i_lora.py
```

### 进阶配置

在配置中，也可以更换采样器和VAE:
```python
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from hcpdiff.diffusion.sampler import DiffusersSampler

wrapper=SD15Wrapper.from_pretrained(
    _partial_=True,
    models=SD15_auto_loader(
        ckpt_path=base_model,
        vae=AutoencoderKL.from_pretrained('any3/vae'), # 更换VAE
        noise_sampler=DiffusersSampler( # 更换采样器
            DPMSolverMultistepScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule='scaled_linear',
                algorithm_type='sde-dpmsolver++',
                use_karras_sigmas=True,
            )
        ),
        _partial_=True
    ),
),
```
