# 模型训练

HCP-Diffusion可以通过```.yaml```配置文件，配置各种训练阶段可能会用到的组件。
包括模型结构，训练参数和方式，数据集配置等。

## 训练基础配置
与训练相关的基础配置文件与示例在```cfgs/train```目录中。所有的训练配置文件都应
同时继承```train_base.yaml```与```tuning_base.yaml```.
```train_base.yaml```中定义了训练阶段需要使用的各种超参数，以及数据集的相关配置。
```tuning_base.yaml```中则定义了训练阶段的模型结构和训练参数，哪些模型参数和embedding需要被训练，哪些层需要以怎样的方式添加lora。

配置文件中的值，可以在cli中修改:
```bash
accelerate launch -m hcpdiff.train_ac --cfg cfgs/train/配置文件.yaml data.dataset1.batch_size=2 seed=1919810
```

## 数据集配置

可以定义多个并行数据集，每个数据集都可以有多个数据源。每步训练会从所有数据集中各抽取一个batch一起训练。
每个数据集中所有数据源会有该数据集的bucket统一处理，并按顺序迭代。 [详细说明](cfg.md#%E6%95%B0%E6%8D%AE%E9%9B%86%E8%AE%BE%E7%BD%AE)

如果已有针对每个图片的```.txt```标注，可以通过下面的命令转换成```.json```标注:
```yaml
python -m hcpdiff.tools.convert_caption_txt2json --data_root 数据集路径
```

### 添加正则化数据集
正则化数据集可以用于DreamBooth，或是让模型学习一些自己生成的图像，保留原有生成能力。

可以从prompt数据库中随机抽取，生成图像，作为这一部分数据。
```bash
python -m hcpdiff.tools.gen_from_ptlist --model 预训练模型 --prompt_file prompt数据库.parquet --out_dir 图像输出路径
```

prompt数据库可以从这里选一个: [prompt数据库](https://huggingface.co/datasets/7eu7d7/HCP-Diffusion-datas/tree/main)

## Bucket
Bucket可以将图像分组排列组合，把具有相同特性的图像放入同一个batch中。

支持的Bucket如下:
+ FixedBucket: 将所有图像缩放并裁剪到给定的相同尺寸。
+ RatioBucket (ARB): 将图像按宽高比分组，各个batch可以有不同宽高比。减少图像裁剪带来的损耗。
    + from ratios: 根据给定的宽高比范围，自动筛选出和目标尺寸最接近的n个不同宽高比的bucket。
    + from_images: 根据训练用到的图像自动对宽高比进行聚类，选出与目标尺寸最接近的n个bucket。

## prompt模板使用 (配合text_transforms)
prompt模板可以在训练阶段将其中的占位符替换成指定的文本。
例如一个prompt模板: 

```a photo of a {pt1} on the {pt2}, {caption}```

其中的```{pt1}```和```{pt2}```会被```text_transforms```中定义的```TemplateFill```替换为指定的词，
这个词可以是自定义的embedding(可以占多个词的位置)，也可以是模型原有的词。
比如定义如下```text_transforms```:
```yaml
text_transforms:
    _target_: torchvision.transforms.Compose
    transforms:
      - _target_: hcpdiff.utils.caption_tools.TemplateFill
        word_names:
          pt1: my-cat # A custom embedding
          pt2: sofa
```
在训练阶段会将```{pt1}```替换为```my-cat```对应的embedding，将```{pt2}```替换为```sofa```.
```{caption}```则会被替换为图像对应的描述，如果没有定义该图像的描述，则这部分为空。

## Fine-tuning训练配置
当前Fine-tuning支持训练unet和text-encoder。可以单独训练模型的一部分，也可以为各个层分配不同的学习率。
格式如下:
```yaml
unet:
  - # group1
    lr: 1e-6 #这一组中所有层的学习率
    layers: #训练哪些层
      # 训练down_blocks的第0号子模块中的所有层
      - 'down_blocks.0'
      # 支持正则表达式，以"re:"开头，训练所有resnet模块中的GroupNorm层
      - 're:.*\.resnets.*\.norm?'
  - # group2
    lr: 3e-6 #这一组中所有层的学习率
    layers:
      # 训练所有的CrossAttention模块
      - 're:.*\.attn.?$'

text_encoder: ...
```
描述训练哪些层所用名称与```model.named_modules()```中的命名一致。

## Prompt-tuning训练配置
prompt-tuning训练word embedding，一个word embedding可以占多个词的位置.

首先需要创建自定义word：
```bash
python -m hcpdiff.tools.create_embedding 预训练模型路径 word名称 一个word占几个词 [--init_text 初始化单词]
# 随机初始化 --init_text *[标准差, word长度]
# 部分随机 --init_text cat, *[标准差, word长度], tail
```

训练哪些词在tokenizer_pt中配置：
```yaml
tokenizer_pt:
  emb_dir: 'embs/' #自定义word目录
  replace: False #训练后是否替换原有word
  train: 
    - {name: pt1, lr: 0.003}
    - {name: pt2, lr: 0.005}
```

## Lora训练配置
当前Lora支持在unet和text-encoder中任意```Linear```和```Conv2d```层中添加。
配置方式与Fine-tuning类似:
```yaml
lora_unet:
  -
    lr: 1e-4
    rank: 2 # Lora模块的rank值
    dropout: 0.0
    alpha: 1.0 #输出=base_model + alpha*lora，如果为1.0则实际scale为1/rank
    svd_init: False #用宿主层的svd分解结果初始化lora模型
    layers:
      - 're:.*\.attn.?$'
  -
    lr: 5e-5
    rank: 0.1 #如果rank为浮点数，则实际rank=对应层输出通道数*rank
    # dropout, alpha, svd_init可以省略，用默认值
    layers:
      - 're:.*\.ff\.net\.0$'

lora_text_encoder: ...
```

## attention mask (可选)

![](../imgs/att_map.webp)

在训练图像较少的情况下，模型难以归纳出什么是重要的特征。所以可以通过添加attention mask，让模型训练阶段更多或更少关注一部分特征。
如上图所示。

attention mask和原始图像应放置于不同文件夹中，并且有着相同的文件名。

attention mask是一个灰度图，其亮度值与注意力倍率如下图所示。

| 亮度  | 0% | 25% | 50%  | 75%  | 100% |
|-----|----|-----|------|------|------|
| 倍率  | 0% | 50% | 100% | 300% | 500% |

## CLIP skip
有些模型在训练阶段会跳过几个CLIP的block，通过设置```model.clip_skip```参数可以跳过CLIP几个block，默认为0(与webui中的clip skip=1等价)，不跳过任何层。