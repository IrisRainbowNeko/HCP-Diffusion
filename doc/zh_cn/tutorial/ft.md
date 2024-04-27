# 模型微调训练指南

这个指南将会教你如何微调一个模型。以预训练底模为基础，使用你的数据集进行进一步训练。

这个教程过程总共分为以下几个部分：
* 准备数据集
* 训练模型
* 生成图像
* 与sd-webui格式转换

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

文本标注支持多种格式，比如`txt`格式的标注，与对应的图片文件名相同，只是后缀不同，比如`1.txt`，`2.txt`等。
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

也可以使用json格式的标注，所有图片的标注放在一个json文件中，格式如下：
```json
{
    "1": "标注1",
    "2": "标注2",
    "3": "标注3",
    ......
}
```

## 模型训练

准备好数据集后，我们可以开始训练模型了。
推荐安装`tensorboard`以更清晰的查看训练进度。
```bash
pip install tensorboard
```

### 训练配置文件填写

在`cfgs/train/examples`中有提供训练相关的模板文件，其中`fine-tuning.yaml`提供了模型微调训练的模板，我们可以继承这个模板文件来进行训练。

模板默认只微调`U-Net`部分。

在`cfgs/train/`中新建一个配置文件，比如`ft1.yaml`：
```yaml
# 继承 fine-tuning.yaml
_base_:
  - cfgs/train/examples/fine-tuning.yaml


unet:
  -
    # 修改学习率
    lr: 1e-6
    layers:
      - '' # 微调unet所有层

train:
  # 设置训练轮数
  train_epochs: 20

model:
  # 底模路径，需要diffusers格式，支持huggingface在线模型
  pretrained_model_name_or_path: 'runwayml/stable-diffusion-v1-5'
  # 句子长度拓展倍数，1倍最长75个词，如果标注很长可以适当调大
  tokenizer_repeats: 1

data:
  dataset1:
    # 每一批的图像数量，根据显存大小调整
    batch_size: 4

    source:
      data_source1:
        # 训练图片路径
        img_root: 'train_data/'
        # 图片标注路径，null则表示没有标注
        # 可以直接填写标注文件所在文件夹，或填写具体的标注文件路径
        caption_file: 'train_data/'
        
        # 设置lora的触发词
        word_names:
          pt1: pt-cat1

    bucket:
      # 预期训练分辨率
      target_area: ${hcp.eval:"512*768"}
      # 按宽高比分组数量，不要超过训练图片数量
      num_bucket: 5
```

配置文件中的参数和路径按实际需求填写。

### 训练

对于单个GPU的运行环境（多卡环境与之类似，详见README），我们可以执行以下的命令进行训练

```shell
accelerate launch -m hcpdiff.train_ac_single \
    --cfg cfgs/train/ft1.yaml 
```

也可以在命令中直接修改参数，如修改学习率和训练轮数：
```shell
accelerate launch -m hcpdiff.train_ac_single \
    --cfg cfgs/train/ft1.yaml \
    train.train_epochs=10 # 修改训练轮数
```

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
```yaml
lora_unet:
  -
    lr: 1e-6
    layers:
      # k,v层和ff层 (使用正则表达式指定层的名称)
      - 're:.*\.to_k$'
      - 're:.*\.to_v$'
      - 're:.*\.ff$'
  -
    lr: 1e-5
    layers:
      - 're:.*resnets$'
```

## 配合DreamBooth微调

如果要配合DreamBooth微调，则需要准备一个正则化数据集，并在配置文件中指定正则化数据集的参数:
```yaml
data:
  dataset1:
    batch_size: 4
    cache_latents: True

    source:
      data_source1:
        img_root: 'train_data/'
        caption_file: 'train_data'

        # DreamBooth的触发词
        word_names:
          pt1: sks
          class: dog
    bucket:
      _target_: hcpdiff.data.bucket.RatioBucket.from_files # aspect ratio bucket
      target_area: ${hcp.eval:"512*512"}
      num_bucket: 1

  dataset_class:
    batch_size: 1
    cache_latents: True
    loss_weight: 1.0

    source:
      data_source1:
        img_root: 'da_data/'
        caption_file: null

        # DreamBooth的触发词，正则化数据集不保留pt1
        word_names:
          class: dog
    bucket:
      _target_: hcpdiff.data.bucket.FixedBucket
      target_size: [512, 512]
```

## 使用微调的模型生成图像

训练完毕后，我们可以使用微调的模型生成图像。这里只简单使用封装好的模板配置文件，更详细的生成配置文件编写请参考[生成配置文件编写](../user_guides/infer.md)。

生成图像
```shell
python -m hcpdiff.visualizer \
    --cfg cfgs/infer/t2i_part.yaml \
    pretrained_model='runwayml/stable-diffusion-v1-5' \
    part_path=exps/2023-07-26-01-05-35/ckpts/unet-1000.safetensors \
    prompt='masterpiece, best quality, 1girl, solo, {surtr_arknights-1000:1.2}'
```

### 进阶配置
可以新建一个配置文件，继承`cfgs/infer/t2i_ft1.yaml`，
```yaml
_base_:
  - cfgs/infer/t2i_part.yaml

# 修改图像生成参数
infer_args:
  width: 768
  height: 768
  guidance_scale: 9.0
  num_inference_steps: 25

new_components:
  # 替换采样器 (DPM++ karras)
  scheduler:
    _target_: diffusers.DPMSolverMultistepScheduler # change Sampler
    beta_start: 0.00085
    beta_end: 0.012
    beta_schedule: 'scaled_linear'
    algorithm_type: 'dpmsolver++'
    use_karras_sigmas: True
    
  # 替换VAE
  vae:
    _target_: diffusers.AutoencoderKL.from_pretrained
    pretrained_model_name_or_path: 'any3.0/vae' # path to vae model
```

## 模型格式转换

如果需要在a1111的sd-webui中使用HCP微调的，需要进行格式转换。

首先保存模型为diffusers格式:
```shell
python -m hcpdiff.visualizer \
    --cfg cfgs/infer/t2i_part.yaml \
    pretrained_model='runwayml/stable-diffusion-v1-5' \
    part_path=exps/2023-07-26-01-05-35/ckpts/unet-1000.safetensors \
    save_model.path=ckpts/ft1-1000
```
模型会保存在`ckpts/ft1-1000`文件夹中。

然后转换成webui格式:
```shell
python -m hcpdiff.tools.sd2diffusers \
    --model_path ckpts/ft1-1000 \
    --save_path ft1-1000.safetensors \
    --use_safetensors
```