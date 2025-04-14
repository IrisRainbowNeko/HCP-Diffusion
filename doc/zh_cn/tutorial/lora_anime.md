# lora训练指南(动漫角色)

本部分将对二次元动漫角色的训练进行介绍。

## 流程与原理

对于这一任务，HCP-Diffusion的作者[IrisRainbowNeko](https://github.com/IrisRainbowNeko)所推荐的方法是，将一个embedding模型和Lora模型一同进行训练，并在实际进行推理（即生成二次元动漫角色的图片）的时候同时使用embedding模型和Lora模型，即可实现所需的效果，并且由于将触发词固化在了embedding模型中，所以将会取得比传统Lora更稳定的效果。

由此，这一训练过程总共分为以下几步：
* 准备数据集
* 创建embedding
* 训练模型
* 模型推理
* sd-webui格式储存

## 准备数据集

首先第一步是准备数据集，我们需要准备若干拥有完全一样尺寸的图片（为确保训练质量建议使用png格式），并且为每一张图片打上对应的文本标签（使用txt格式），最终形成类似如下的格式（此处该数据集保存在`/data/surtr_dataset`，且图片的尺寸均为512x704）

```text
/data/surtr_dataset
├── 000834cb567b675eb0904436b2d6dabdb5b09493.png
├── 000834cb567b675eb0904436b2d6dabdb5b09493.txt
├── 0095c8ff0ccaf9ab30c705d9babef91800042497.png
├── 0095c8ff0ccaf9ab30c705d9babef91800042497.txt
├── 00e73bb48d5a2dded1464a433d619f901ee07d6a.png
├── 00e73bb48d5a2dded1464a433d619f901ee07d6a.txt
├── ......
```

```{tip}
一种推荐的方式，是使用[waifuc](https://github.com/deepghs/waifuc)项目，输入角色的英文名，一键完成角色图片在十多个图片网站（pixiv、danbooru、zerochan等）上的爬取、清洗、处理、打标等一系列环节。
```

## 创建embedding

为了提高出图时触发词的稳定性，本文介绍的训练方式将需要用到一个embedding（和Texture Inversion类似），可以近似的理解为一个embedding代表一个关键词。

首先我们使用以下的命令创建embedding

```shell
python -m hcpdiff.tools.create_embedding 预训练模型路径 word名称 一个word占几个词 [--init_text 初始化单词]
```

例如对于我们需要训练的角色史尔特尔（关键词名称为：`surtr_arknights`），我们可以这样创建embedding

```shell
python -m hcpdiff.tools.create_embedding deepghs/animefull-latest surtr_arknights 4
```

此时，在`embs`路径下，将包含一个`surtr_arknights.pt`文件。


## 模型训练

完成了上述准备后，我们可以开始进行训练。

首先我们需要安装Tensorboard以实时查看训练进度

```shell
pip install tensorboard
```

::::{tab-set}
:::{tab-item} 单GPU训练

对于单个GPU的运行环境，我们可以执行以下的命令进行训练
```shell
hcp_train_1gpu --cfg cfgs/train/py/examples/lora_preview.py \
    model.wrapper.models.ckpt_path=deepghs/animefull-latest \  # 底模
    data_train.dataset1.handler.word_names.pt1=surtr_arknights \  # 触发词
    data_train.dataset1.source.data_source1.img_root=/data/surtr_dataset  # 数据集
```

:::

:::{tab-item} 多GPU训练

对于多个GPU的运行环境，需要在`cfgs/launcher/multi.yaml`中指定训练用到的GPU id和GPU数量，然后执行以下的命令进行训练
```shell
hcp_train --cfg cfgs/train/py/examples/lora_preview.py \
    model.wrapper.models.ckpt_path=deepghs/animefull-latest \  # 底模
    data_train.dataset1.handler.word_names.pt1=surtr_arknights \  # 触发词
    data_train.dataset1.source.data_source1.img_root=/data/surtr_dataset  # 数据集
```

:::
::::

```{note}
其中：
* `data_train.dataset1.handler.word_names.pt1`填写待训练的角色的名称，应于上个部分中创建的embedding的名称一致，此处为`surtr_arknights`。
* `data_train.dataset1.source.data_source1.img_root`为数据集的存储路径，此处填写`/data/surtr_dataset`。
* 【可选】`exp_dir`为实验数据的保存路径，默认情况下将使用在`exps`路径下生成一个用当前时间命名的子路径，如`exps/2023-07-26-01-05-35`。
* 【可选】`train.train_steps`为训练总步数，默认值为`1000`。
* 【可选】`train.save_step`为训练时的保存步数间隔，默认值为`100`，即每训练100步保存一次模型。
* 【可选】`model.wrapper.models.ckpt_path`为训练时基于的扩散模型，默认值为`deepghs/animefull-latest`，即为NovelAI官方泄露的模型，大小约为7G。该模型为动漫角色训练的泛用模型，训练时将自动从HuggingFace仓库中下载到本地并用于训练。
```

训练完毕后，你将得到这样一个实验数据路径

```text
exps/2023-07-26-01-05-35
├── cfg.yaml
├── ckpts
│   ├── surtr_arknights-1000.pt
│   ├── surtr_arknights-100.pt
│   ├── surtr_arknights-200.pt
│   ├── surtr_arknights-300.pt
│   ├── surtr_arknights-400.pt
│   ├── surtr_arknights-500.pt
│   ├── surtr_arknights-600.pt
│   ├── surtr_arknights-700.pt
│   ├── surtr_arknights-800.pt
│   ├── surtr_arknights-900.pt
│   ├── text_encoder-1000.safetensors
│   ├── text_encoder-100.safetensors
│   ├── text_encoder-200.safetensors
│   ├── text_encoder-300.safetensors
│   ├── text_encoder-400.safetensors
│   ├── text_encoder-500.safetensors
│   ├── text_encoder-600.safetensors
│   ├── text_encoder-700.safetensors
│   ├── text_encoder-800.safetensors
│   ├── text_encoder-900.safetensors
│   ├── unet-1000.safetensors
│   ├── unet-100.safetensors
│   ├── unet-200.safetensors
│   ├── unet-300.safetensors
│   ├── unet-400.safetensors
│   ├── unet-500.safetensors
│   ├── unet-600.safetensors
│   ├── unet-700.safetensors
│   ├── unet-800.safetensors
│   └── unet-900.safetensors
├── tblog
│   └── events.out.tfevents.1690346085.myenvironment.210494.0
└── train.log
```

```{note}
其中：
* `surtr_arknights-xxx.pt`为训练所得的embedding。
* `text_encoder-xxx.safetensors`和`unet-xxx.safetensors`为训练所得的Lora模型。（注：在HCP-Diffusion框架下，Lora模型将被分为两部分，如果需要转换为webui支持的Lora模型格式，请参考[模型文件格式说明](../user_guides/model_format.md)）。
```

## 模型推理

完成训练后，我们使用之前训练所得的模型生成图片

::::{tab-set}
:::{tab-item} 使用配置文件

将`cfgs/workflow/easy/t2i_lora.py`复制一份`cfgs/workflow/easy/t2i_lora_surtr.py`，并修改其中的配置

```python
@neko_cfg
def make_cfg():
    return SD15_t2i_lora(
        pretrained_model='stablediffusionapi/anything-v5', # 替换预训练底模
        lora_info=[
            ('exps/2023-07-26-01-05-35/lora1-1000.safetensors', 0.8), # (lora模型路径, lora权重)
        ],
        prompt='masterpiece, best quality, 1girl, solo, {surtr_arknights-1000:1.2}', # 生成图像的prompt
        bs=4,
        width=512,
        height=768,
        guidance_scale=7.5
    )
```

```{note}
其中：
* `lora_info`中添加了训练所得的Lora模型路径和权重，权重值可以根据需要进行调整。
* `prompt`为生成图片时的提示词。请注意，使用embedding中的触发词时，格式应当为`character_name-xxxx`，其中`xxxx`为步数，该值应当与`model_steps`保持一致。在本例子中，即为`surtr_arknights-1000`。
* 【可选】`negative_prompt`为生成图片时的负面提示词。默认值为一条通用的负面提示词。
* 【可选】`N_repeats`代表提示词的容量。默认值为`1`，当提示词较长且由此导致出现报错后，可以提高该值。
* 【可选】`pretrained_model`为生成图片所使用的基模型。默认值为`stablediffusionapi/anything-v5`，该模型在实际的动漫图片生成中拥有比`deepghs/animefull-latest`更好的性能。
* 【可选】`width`为生成图片的宽度，应当为8的整倍数。默认值为`512`。
* 【可选】`height`为生成图片的高度，应当为8的整倍数。默认值为`768`。
* 【可选】`guidance_scale`为生成图片时的scale，该值越高，提示词对图片的控制力将越强，生成的图片也将越趋同。默认值为`7.5`。
* 【可选】`N_steps`为生成图片时的步数。默认值为`20`。
* 【可选】`bs`为生成图片的数量。默认值为`4`。
* 【可选】`seed`为生成图片时的随机种子，当使用同一个种子，且其他配置也相同时，生成的图片将是完全确定的。当`seed`未指定时，将随机使用一个随机种子，具体值在生成图片对应的python配置文件中可以找到。
* 【可选】`save_root`图片文件的导出路径。默认值为`output_pipe/`。
```

使用配置文件生成图像
```shell
hcp_run --cfg cfgs/workflow/easy/t2i_lora_surtr.yaml
```

:::

:::{tab-item} 使用封装的配置，命令行输入参数
```shell
hcp_run --cfg cfgs/workflow/easy/t2i_lora_cli.yaml \
    lora_path=exps/2023-07-26-01-05-35/lora1-1000.safetensors \
    prompt='masterpiece, best quality, 1girl, solo, {surtr_arknights-1000:1.2}'
```
:::
::::


运行完毕后，将在`output`路径下生成一个png图片和一个python配置文件，分别为生成的图片和生成时的具体配置信息。一种可能的图片如下所示（因seed为随机取值，因此图片可能与下图不同，仅供参考）：

![surtr_arknight_sample](../../imgs/surtr_arknights_sample.png)


## 模型在webui中使用

如果需要在a1111的webui中使用训练的LoRA模型，需要保存为webui支持的格式。具体方法见 [lora训练指南](./lora.md)


webui版本的Lora模型文件将被储存在`exps/2023-07-26-01-05-35/ckpts/`文件夹。

如果你需要将文件发布在civitai.com，则将以下文件上传即可：
* `exps/2023-07-26-01-05-35/ckpts/lora_webui-1000.safetensors`，Lora模型文件
* `exps/2023-07-26-01-05-35/ckpts/surtr_arknights-1000.pt`，embedding触发词文件

在webui上，只要同时使用这两个模型，即可画出你的二次元老婆啦~~~

