# lora训练指南

这个指南将会教你如何训练一个Lora模型，为SD底模添加一些新的知识。

这个教程过程总共分为以下几个部分：
* 准备数据集
* 训练模型
* 使用lora生成图像
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

在`cfgs/train/examples`中有提供训练相关的模板文件，其中`lora_conventional.yaml`提供了lora模型训练的模板，我们可以继承这个模板文件来进行训练。

模板默认只为`U-Net`部分添加lora，不在`text encoder`部分添加lora。

在`cfgs/train/`中新建一个配置文件，比如`lora1.yaml`：
```yaml
# 继承lora_conventional.yaml
_base_:
  - cfgs/train/examples/lora_conventional.yaml

# 修改学习率
unet_lr: 0.0006

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
    --cfg cfgs/train/lora1.yaml 
```

也可以在命令中直接修改参数，如修改学习率和训练轮数：
```shell
accelerate launch -m hcpdiff.train_ac_single \
    --cfg cfgs/train/lora1.yaml \
    unet_lr=0.0002 \ # 修改学习率
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

自定义lora参数和添加lora的层:
```yaml
lora_unet:
  -
    lr: ${unet_lr}
    rank: 16 # lora的维度
    alpha: 2 # lora的比例系数
    layers:
      # k,v层和ff层 (使用正则表达式指定层的名称)
      - 're:.*\.to_k$'
      - 're:.*\.to_v$'
      - 're:.*\.ff$'

lora_text_encoder: # 添加lora到text encoder
  - lr: 2e-5
    rank: 2
    dropout: 0.1
    layers:
      - 're:.*self_attn$' # 注意力层
      - 're:.*mlp$' # mlp层
```

## 使用lora生成图像

训练完毕后，我们可以使用lora生成图像。这里只简单使用封装好的模板配置文件，更详细的生成配置文件编写请参考[生成配置文件编写](../user_guides/infer.md)。
```shell
python -m hcpdiff.visualizer \
    --cfg cfgs/infer/t2i_lora.yaml \
    pretrained_model='runwayml/stable-diffusion-v1-5' \
    lora_path=exps/2023-07-26-01-05-35/ckpts/unet-1000.safetensors \
    lora_alpha=0.8 \
    prompt='masterpiece, best quality, 1girl, solo, {surtr_arknights-1000:1.2}'
```

### 进阶配置
可以新建一个配置文件，继承`cfgs/infer/t2i_lora.yaml`，
```yaml
_base_:
  - cfgs/infer/t2i_lora.yaml

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

如果需要在a1111的sd-webui中使用HCP训练的lora，需要进行格式转换。
```shell
python -m hcpdiff.tools.lora_convert --to_webui \
    --lora_path unet-xxxx.safetensors \ # unet lora路径
    --lora_path_TE text_encoder-xxxx.safetensors \ # 没有text encoder就不加这项
    --dump_path lora-xxxx.safetensors \ # 输出路径
    --auto_scale_alpha # 现有webui模型没有alpha自动缩放，需要转换
```