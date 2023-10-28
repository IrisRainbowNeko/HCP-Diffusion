# 配置文件说明

这部分主要介绍```cfgs/train/train_base.yaml```中的训练参数设置。

配置文件的语法为```yaml```格式，支持```OmegaConf```和```hydra```的拓展语法。

## 训练参数

```yaml
train:
  #梯度累积步数
  #总batch_size = 各数据集batch_size总和 * 梯度累积步数 * GPU数量
  gradient_accumulation_steps: 1
  
  workers: 4 #并行加载数据的进程数，可以根据cpu核心数量调整
  max_grad_norm: 1.0 #梯度剪裁，防止梯度爆炸
  set_grads_to_none: False #重置梯度时是否设置为None
  save_step: 100 # 保存模型的步数间隔
  
  # DreamArtist中的CFG强度，1.0表示不使用DreamArtist
  # 支持动态CFG，随diffusion的时间步动态变化
  # 格式为：下限-上限:激活函数。激活函数默认线性，cos为cos函数的0-π/2区间，cos2为cos函数的π/2-π区间
  cfg_scale: '1.0' 

  resume: # 接着之前的训练，如果为null则从头训练
    ckpt_path:
      unet: [] # 所有unet权重文件的路径
      TE: [] # 所有text-encoder权重文件的路径
      words: {} # 所有自定义word权重文件的路径
    start_step: 0 # 之前的训练结束的步数

  loss: # 损失函数配置
    criterion:
      # 这里使用 hydra.utils.instantiate 的语法定义
      # 所有具有 _target_ 属性的模块都会被实例化为对应的python对象
      _target_: torch.nn.MSELoss # 损失函数对应的类别
      _partial_: True
      reduction: 'none' # 不在内部做平均，以支持attention mask
    # data_class部分数据对应的loss的权重
    # 保持 data.batch_size/(data_class.batch_size*prior_loss_weight) = 4/1可以得到较好的效果
    type: 'eps'

  optimizer: # 模型参数部分优化器
    _target_: torch.optim.AdamW # 优化器的类路径
    _partial_: True
    weight_decay: 1e-3 # weight_decay用于正则化，提升效果
    
  optimizer_pt: # 单词参数优化器
    _target_: torch.optim.AdamW
    _partial_: True
    weight_decay: 5e-4

  scale_lr: True # 是否按总batch size自动缩放学习率
  scheduler: # 学习率调整方案，各种方案见下一节
    name: 'one_cycle' # scheduler类型
    num_warmup_steps: 200 # 学习率逐渐变大的步数
    num_training_steps: 1000 # 训练总步数
    scheduler_kwargs: {} # 其他scheduler需要的参数

  scale_lr_pt: True # 是否按总batch size自动缩放单词训练的学习率
  scheduler_pt: ${.scheduler} # 单词训练的学习率调整方案。OmegaConf语法，与上面的scheduler内容一致
```

## 学习率调整方案

![](../imgs/lr.webp)

上图显示了各种学习率调整策略随步数的变化，推荐使用```one_cycle```或```constant_with_warmup```.
上升部分通过```num_warmup_steps```设置，总步数通过```num_training_steps```设置。

```one_cycle```还可以调整下面两个参数，写到```scheduler_kwargs```中:
+ div_factor: 最大lr/起始lr
+ final_div_factor: 最大lr/结束lr

## 模型参数

```yaml
model:
  revision: null # 预训练模型的版本
  pretrained_model_name_or_path: null # 预训练模型的路径或名称
  tokenizer_name: null # 可以单独指定使用的tokenizer
  tokenizer_repeats: 1 # 将句子长度拓展N倍，如果caption超出上限可以增加tokenizer_repeats
  enable_xformers: True # 开启xformers优化
  gradient_checkpointing: True # 开启优化，节省显存
  ema_unet: 0 # unet部分ema模型的超参数，0为关闭。通常设置为0.9995
  ema_text_encoder: 0 # text-encoder部分ema模型的超参数
  clip_skip: 0 # 跳过text-encoder最后N层，值为0与webui的clip_skip=1一致
  clip_final_norm: True # 使用CLIP的最后一个正则化层
```

### 自定义插件
配置文件中的```plugin_unet```和```plugin_TE```模块中可以添加自定义插件，格式如下:
```yaml
# 此处以controlnet和loha的定义为例
plugin_unet:
  controlnet1: # 插件的名称，每个插件的名称都不能相同
    _target_: hcpdiff.models.controlnet.ControlNetPlugin # 插件类的路径，也可以通过classmethod创建。
    _partial_: True # 必须添加的属性
    train: True # 是否训练这个插件，默认为True
    lr: 1e-4 # 学习率
    # controlnet为MultiPluginBlock类型，可以定义多个输出层(from_layers)和输出层(to_layers)
    from_layers:
      - 'pre_hook:' # pre_hook前缀表示hook到该层forward前，如果没有就hook到forward之后
      - 'pre_hook:conv_in' # 在这里运行自己的forward，让forward可以在autocast内部
    to_layers:
      - 'down_blocks.0'
      - 'down_blocks.1'
      - 'down_blocks.2'
      - 'down_blocks.3'
      - 'mid_block'
      - 'pre_hook:up_blocks.3.resnets.2'
  loha1: # 插件的名称
    _target_: hcpdiff.models.lora_layers.LohaLayer.wrap_model
    _partial_: True
    train: False
    #定义插件需要的参数
    rank: 8
    dropout: 0.15
    rank_groups: 2
    
    # LohaLayer为SinglePluginBlock类型，可以通过正则表达式定义多个层
    layers:
      - 're:.*\.attn.?$'
      - 're:.*\.ff\.net\.0$'
```

插件总共有三种类型:
+ SinglePluginBlock: 单层插件，根据该层输入改变输出，比如lora系列。支持正则表达式(```re:```前缀)定义插入层，
  不支持```pre_hook:```前缀。
+ PluginBlock: 输入层和输出层都只有一个，比如定义残差连接。支持正则表达式(```re:```前缀)定义插入层，
  输入输出层都支持```pre_hook:```前缀。
+ MultiPluginBlock: 输入层和输出层都可以有多个，比如controlnet。不支持正则表达式(```re:```前缀)，
  输入输出层都支持```pre_hook:```前缀。

所有自定义插件都需要继承上面某一种插件基类。

## 数据集设置

可以定义多个并行数据集，每个数据集都可以有多个数据源。每步训练会从所有数据集中各抽取一个batch一起训练。
每个数据集中所有数据源会有该数据集的bucket统一处理，并按顺序迭代。

```yaml
data:
  # 可以定义多个并行数据集，每步训练会从所有数据集中各抽取一个batch一起训练
  dataset1:
    _target_: hcpdiff.data.TextImagePairDataset # 数据集类路径
    _partial_: True # 必须加，为了在后续添加额外参数
    batch_size: 4 # 这一部分数据集的batch_size
    cache_latents: True # 是否预先将图像用VAE编码，可以加快训练速度
    att_mask_encode: False # 是否对attention_mask应用VAE中的self-attention
    loss_weight: 1.0 # 这部分数据集在计算loss时的权重
    
    # 定义一个所有数据源通用的图像变换，具体细节参考 torchvision.transforms
    image_transforms:
      _target_: torchvision.transforms.Compose # "_target_" for hydra.utils.instantiate
      transforms:
        - _target_: torchvision.transforms.ToTensor
        - _target_: torchvision.transforms.Normalize
          _args_: [[0.5], [0.5]]
    
    # 数据来源，所有源的图像都会被这部分数据集的bucket统一处理，作为一个整体。
    # 每个数据集可以有多个数据源。
    source:
      data_source1: #数据源1
        img_root: 'imgs/train' # 图像文件夹
        # prompt填充模板，填充词在下面的 utils.caption_tools.TemplateFill 中配置
        prompt_template: 'prompt_tuning_template/object.txt'
        caption_file: null # path to image captions (file_words)
        att_mask: null # attention_mask的文件夹路径
        bg_color: [255, 255, 255] # 读取透明图像时的填充背景色
        image_transforms: ${...image_transforms} # 图像增强与预处理
        text_transforms: # 文本增强与预处理
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: hcpdiff.utils.caption_tools.TagShuffle # 按 "," 打乱描述的顺序
            - _target_: hcpdiff.utils.caption_tools.TagDropout # 按 "," 分割描述，随机删除
              p: 0.1 # 删除的概率
            - _target_: hcpdiff.utils.caption_tools.TemplateFill # 填充prompt模板，每次随机从模板文件中抽取一行
              word_names:
                pt1: pt-cat1 # 将模板中的{pt1}替换为 pt-cat1
                class: cat # 将模板中的{class}替换为 cat
      data_source2: ... #数据源2
      data_source3: ... #数据源3
    bucket: # 使用什么样的bucket对图像进行处理和分组
      _target_: hcpdiff.data.bucket.RatioBucket.from_files # 按所有图像的比例自动聚类分组，尽可能避免切图
      # 训练使用的图像尺寸，值为面积
      # 此处使用hydra语法，调用python的eval函数计算面积
      target_area: {_target_: "builtins.eval", _args_: ['512*512']}
      num_bucket: 5 # 分多少个组
  
  dataset_class: # 正则化数据集，与上面的并行
    _target_: hcpdiff.data.TextImagePairDataset
    _partial_: True
    batch_size: 1
    cache_latents: True
    att_mask_encode: False
    loss_weight: 0.8

    source:
      data_source1:
        img_root: 'imgs/db_class'
        prompt_template: 'prompt_tuning_template/object.txt'
        caption_file: null
        att_mask: null
        bg_color: [255, 255, 255] # RGB; for ARGB -> RGB
        image_transforms: ${....dataset1.source.data_source1.image_transforms}
        text_transforms:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: hcpdiff.utils.caption_tools.TagShuffle
            - _target_: hcpdiff.utils.caption_tools.TagDropout
              p: 0.1
            - _target_: hcpdiff.utils.caption_tools.TemplateFill
              word_names:
                class: cat
    bucket:
      _target_: hcpdiff.data.bucket.FixedBucket # 将图像剪裁为固定大小训练
      target_size: [512, 512] # 使用的尺寸
```

## Loss配置

Min-SNR loss:
```yaml
loss:
  criterion:
    # 其余属性会继承 train_base
    _target_: hcpdiff.loss.MinSNRLoss # 损失函数对应的类别
    gamma: 2.0
```

## 其他参数
```yaml
# 继承的父配置文件，该文件的参数基于父文件修改，可以继承多个文件
# 只需要写修改的参数，其他参数用父文件默认值
# 列表会被全部替换，要写全
_base_: [cfgs/train/train_base.yaml, cfgs/train/tuning_base.yaml]

exp_dir: exps/ # 输出文件夹
mixed_precision: 'fp16' # 是否使用半精度训练加速
seed: 114514 # 训练用的随机种子
ckpt_type: 'safetensors' # [torch, safetensors]，存torch格式还是safetensors格式
```