# 模型微调训练指南

这个指南将会教你如何微调一个模型。以预训练底模为基础，使用你的数据集进行进一步训练。

这个教程过程总共分为以下几个部分：
* 准备数据集
* 训练模型
* 生成图像
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

:::::{tab-set}
::::{tab-item} 常规配置文件

在`cfgs/train/py/examples`中有提供训练相关的模板文件，其中`SD_FT.py`提供了SD1.5模型训练的模板，我们可以复制这个模板文件来进行训练。

模板默认只微调`U-Net`部分。

在`cfgs/train/py/`中复制一个配置文件，比如`ft1.py`，其中主要需要修改的配置如下:
```python
from cfgs.workflow import text2img

model_part=CfgWDModelParser([
    dict(
        lr=1e-5, # 模型学习率
        layers=['denoiser'],  # 指定需要训练的层，这里训练U-Net
    )
], weight_decay=1e-2), # weight_decay一般为0.01或者0.001
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
# 添加预览功能
evaluator=HCPPreviewer(_partial_=True,
    interval=100, # 预览间隔
    workflow=text2img, # 使用哪个工作流预览图像
),
```

配置文件中的参数和路径按实际需求填写。在预览图像时，会使用训练中使用的模型。

对于数据集配置:
```python
@neko_cfg
def cfg_data():
    return dict(
        dataset1=TextImagePairDataset(_partial_=True,
            batch_size=4, # 批次大小
            loss_weight=1.0,  # 这个数据集的权重
            source=dict(
                data_source1=Text2ImageSource(
                    img_root= 'imgs/', # 图片路径
                    label_file= '${.img_root}',  # 标注路径，默认与图片一致
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(bucket=RatioBucket), # bucket类型与下面相同
            bucket=RatioBucket.from_files(
                target_area=512*512, # 图片训练分辨率
                num_bucket=6, # 分桶数量，训练集分辨率越多，分桶数量越多
            ),
            cache=VaeCache(bs=4) # 缓存VAE编码，减少显存使用
        )
    )
```
::::
::::{tab-item} 简化配置文件

在`cfgs/train/py/easy`中有提供训练相关的简化模板文件，其中`SD15_FT.py`提供了SD1.5模型微调的模板，我们可以基于这个模板文件来进行训练。

模板默认只训练`U-Net`部分。

```python
from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SD15_finetuning, cfg_data_SD_ARB

@neko_cfg
def make_cfg():
    return SD15_finetuning(
        base_model='Lykon/DreamShaper', # 预训练模型路径
        train_steps=1000, # 训练总步数
        save_step=200, # 保存模型步数间隔
        dataset=dict(
            dataset1=cfg_data_SD_ARB( # 使用ARB使图像尽可能不被剪裁
                img_root='imgs/', # 训练图片路径，标注和图片在同一个文件夹
                batch_size=4, # 批次大小
                resolution=512*512, # 训练分辨率
                num_bucket=4, # 分桶数量，训练集分辨率越多，分桶数量越多
            )
        )
    )
```

:::{dropdown} 添加预览功能
:animate: fade-in

添加简单的配置就可以为模型加入预览功能

```python
from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SD15_finetuning, cfg_data_SD_ARB, SD15_t2i
from hcpdiff.evaluate import HCPPreviewer

@neko_cfg
def make_cfg():
    return dict(
        **SD15_finetuning(
            base_model='Lykon/DreamShaper', # 预训练模型路径
            train_steps=1000, # 训练总步数
            save_step=200, # 保存模型步数间隔
            dataset=dict(
                dataset1=cfg_data_SD_ARB( # 使用ARB使图像尽可能不被剪裁
                    img_root='imgs/', # 训练图片路径，标注和图片在同一个文件夹
                    batch_size=4, # 批次大小
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
                prompt='', # 预览使用的prompt
                seed=42, # 固定随机种子
                
                ## 可选配置
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

在这个配置基础上，还有一些其他配置可以选择:
```python
@neko_cfg
def make_cfg():
    return SD15_lora_train(
            base_model='Lykon/DreamShaper', # 预训练模型路径
            train_steps=1000, # 训练总步数
            save_step=200, # 保存模型步数间隔
            
            lr=1e-4, # 模型学习率
            clip_skip=0, # clip跳过的层数，0表示不跳过
            low_vram=False, # 使用8bit优化器
            
            dataset=dict(
                dataset1=cfg_data_SD_ARB( # 使用ARB使图像尽可能不被剪裁
                    img_root='imgs/', # 训练图片路径，标注和图片在同一个文件夹
                    batch_size=4, # 批次大小
                    resolution=512*512, # 训练分辨率
                    num_bucket=6, # 分桶数量，训练集分辨率越多，分桶数量越多
                )
            )
        )
```

::::
:::::

### 训练

::::{tab-set}
:::{tab-item} 单GPU训练

```bash
hcp_train_1gpu --cfg cfgs/train/py/ft1.py
```

配置文件中的值，可以在cli中修改:
```bash
hcp_train_1gpu --cfg cfgs/train/py/ft1.py train.train_epochs=10 # 修改训练轮数
```
:::

:::{tab-item} 多GPU训练
多GPU训练需要在`cfgs/launcher/multi.yaml`中指定训练用到的GPU id和GPU数量，然后运行:
```bash
hcp_train --cfg cfgs/train/py/ft1.py
```

配置文件中的值，可以在cli中修改:
```bash
hcp_train --cfg cfgs/train/py/ft1.py train.train_epochs=10 # 修改训练轮数
```
:::
::::

训练完毕后，输出的所有结果都会存到一个文件夹中:
```text
exps/2023-07-26-01-05-35
├── cfg.yaml # 配置文件
├── ckpts # lora模型文件
│   ├── unet-100.safetensors
│   ├── unet-200.safetensors
│   ├── ...
├── tblog # tensorboard日志
│   └── events.out.tfevents.1690346085.myenvironment.210494.0
└── train.log
```

### 进阶配置

分层训练，或只练一部分层:
```python
model_part=CfgWDModelParser([
    dict(
        lr=1e-6,
        # k,v层和ff层 (使用正则表达式指定层的名称)
        layers=[
            're:.*\.to_k$'
            're:.*\.to_v$'
            're:.*\.ff$'
        ],
    ),
    dict(
        lr=1e-5,
        layers=[
            're:.*resnets$'
        ],
    )
], weight_decay=1e-2),
```

## 使用DreamBooth微调

如果要配合DreamBooth微调，则需要准备一个正则化数据集，并在配置文件中指定正则化数据集的参数:
```python
@neko_cfg
def cfg_data():
    return dict(
        # 训练数据集
        dataset1=TextImagePairDataset(_partial_=True, batch_size=4, loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root= 'imgs/',
                    label_file= '${.img_root}',  # path to image captions
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(
                bucket=RatioBucket,
                word_names={
                    'pt1': '[V]', # 触发词
                    'class': 'dog' # 训练主体描述
                }
            ),
            bucket=RatioBucket.from_files(
                target_area=512*512,
                num_bucket=6,
            ),
            cache=VaeCache(bs=1)
        ),
        # 正则化数据集
        dataset_class=TextImagePairDataset(_partial_=True, batch_size=1, loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root='imgs_db_class/',
                    label_file='${.img_root}',
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(
                bucket=FixedBucket,
                word_names={'class': 'dog'} # 训练主体描述
            ),
            bucket=FixedBucket(
                target_size=(512, 512),
            ),
            cache=VaeCache(bs=1)
        )
    )
```

```{tip}
$batch\_size \times loss\_weight$ 可以看作数据集的重要度。推荐训练数据集和正则数据集的比例是4:1。
```

## 使用微调的模型生成图像

训练完毕后，我们可以使用微调的模型生成图像。这里只简单使用封装好的模板配置文件，更详细的生成配置文件编写请参考[生成配置文件编写](../user_guides/workflow.md)。

这里可以使用简化的配置文件:
```python
from hcpdiff.easy.cfg import SD15_t2i_parts
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SD15_t2i_parts(
        pretrained_model='Lykon/DreamShaper', # 预训练底模
        parts=['unet-100.safetensors'], # 训练的模型路径
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        bs=4,
        width=512,
        height=512,
        guidance_scale=7.0
    )
```

保持并运行配置文件生成图片:
```shell
hcp_run --cfg cfgs/workflow/t2i.py
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