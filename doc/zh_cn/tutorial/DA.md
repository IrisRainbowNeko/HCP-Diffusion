# DreamArtist++ 使用教程

DreamArtist可以仅用一张图像完成prompt-tuning训练(one-shot prompt-tuning)。
DreamArtist++则在此基础上添加lora模块，大幅提高训练的稳定性和模型的可控性。
仅用一张图像便可以训练出有高泛化性和可控性的lora模型。

注意：

虽然训练阶段同时训练了 **正负word** ，但生成阶段只使用正prompt会得到更好的效果。
```
正: 正word + 正lora
负: 负lora
```

## DreamArtist++
DreamArtist同时训练positive和negative两个分支，每个分支都有其对应的lora和word embedding。
因此在配置文件中需要分别定义positive和negative部分的信息。

```yaml
lora_unet:
  - lr: 1e-4
    rank: 3
    branch: p # positive分支
    layers:
      - 're:.*\.attn.?$'
      #- 're:.*\.ff\.net\.0$' # 增加拟合程度，但有可能减少泛化性和可控性
  - lr: 2e-5 # Low negative unet lr prevents image collapse
    rank: 3
    branch: n # negative分支
    layers:
      - 're:.*\.attn.?$'
      #- 're:.*\.ff\.net\.0$'

lora_text_encoder:
  - lr: 1e-5
    rank: 1
    branch: p
    layers:
      - 're:.*self_attn$'
      - 're:.*mlp$'
  - lr: 1e-5
    rank: 1
    branch: n
    layers:
      - 're:.*self_attn$'
      - 're:.*mlp$'
```

这两个lora分支共享同一个基础模型，需要不同的触发词，各自的触发词在data中定义(需要提前创建word):
```yaml
data:
  text_transforms:
    transforms:
      - _target_: hcpdiff.utils.caption_tools.TemplateFill
        word_names:
          pt1: [my-cat, my-cat-neg] #触发词需要正负一对
```

在没有触发词的时候，模型应该尽可能与原来表现的一致，减少画风污染，因此还可以让模型学一部分自己生成的图像:
```yaml
data_class:
  caption_file: dataset/image_captions.json #训练text-image对
  text_transforms:
    transforms:
      - _target_: hcpdiff.utils.caption_tools.TemplateFill
        word_names:
          pt1: ['', ''] #由于使用了DreamArtist++，所有填充词都需要正负一对
```
生成用的prompt可以从prompt数据库中随机抽取。

prompt数据库可以从这里选一个: [prompt数据库](https://huggingface.co/datasets/7eu7d7/HCP-Diffusion-datas/tree/main)