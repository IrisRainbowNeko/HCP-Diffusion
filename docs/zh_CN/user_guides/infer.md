# 图像生成(推理)

推理阶段同样使用```.yaml```配置文件描述使用哪些训练的组件，以及使用什么参数生成。

## 生成参数

```yaml
pretrained_model: '' # 预训练基础模型
prompt: '' # 生成使用的文本
#负文本，可以将不要的特征排除掉
neg_prompt: 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
out_dir: 'output/' # 图像输出文件夹
emb_dir: 'embs/' # 自定义单词(embedding)文件夹
N_repeat: 1 # 句子长度扩展倍数
clip_skip: 1
bs: 4 # batch_size
num: 1 # 总图像数=bs*num
seed: null # 随机种子
fp16: True # 半精度推理速度更快，显存更少

condition: null # img2img和contorlnet

save:
  save_cfg: True # 是否一起储存配置文件
  image_type: png # 储存图像格式
  quality: 95 # 储存图像压缩质量

infer_args:
  width: 512 # 图像宽度
  height: 512 # 图像高度
  guidance_scale: 7.5 # CFG scale大小

new_components: {} # 替换模型的组件 Sampler，VAE 等

merge: null # 加载模型和lora等插件
```

替换Sampler:
```yaml
new_components:
  scheduler:
    _target_: diffusers.EulerAncestralDiscreteScheduler # change Sampler
    beta_start: 0.00085
    beta_end: 0.012
    beta_schedule: 'scaled_linear'
```

替换VAE:
```yaml
new_components:
  vae:
    _target_: diffusers.AutoencoderKL.from_pretrained
    pretrained_model_name_or_path: 'any3.0/vae' # path to vae model
```

## 加载训练的模型
可以指定多个模型进行分层融合，或者添加多个lora。

```yaml
merge:
  exp_dir: '2023-04-03-10-10-36'
  
  group1: #可以同时加载多个组，有不同的配置参数。
    type: 'unet' #加载到unet上还是text-encoder上
    base_model_alpha: 1.0 # base model weight to merge with lora or part
    lora: #一个组可以加载多个lora模型
      - path: 'exps/${....exp_dir}/ckpts/unet-600.ckpt'
        alpha: 0.7 # base_model*base_model_alpha + lora*alpha
        layers: 'all' #指定加载哪些层，"all"为全部层
        mask: [0.5, 1] #lora的batch_mask，[0.5, 1]对应DreamArtist++的positive分支
      - path: 'exps/${....exp_dir}/ckpts/unet-neg-600.ckpt'
        alpha: 0.55
        layers: 'all'
        mask: [0, 0.5] #[0, 0.5]对应DreamArtist++的negative分支
    part: null
  
  group2:
    type: 'TE' #加载到text-encoder上
    base_model_alpha: 1.0 # base model weight to merge with lora or part
    lora:
      - path: 'exps/${....exp_dir}/ckpts/text_encoder-600.ckpt'
        alpha: 0.7
        layers: 'all'
        mask: [0.5, 1]
      - path: 'exps/${....exp_dir}/ckpts/text_encoder-neg-600.ckpt'
        alpha: 0.55
        layers: 'all'
        mask: [0, 0.5]
    part: null
  
  group3:
    type: 'unet'
    base_model_alpha: 0.0 # 替换基础模型的层
    part: 
      - path: 'exps/${....exp_dir}/ckpts/unet_ft.ckpt'
        alpha: 1.0 # 替换基础的模型层
        layers: 'all'
```

如果使用DreamArtist方法，需要在prompt和negative_prompt中都加入对应的触发词```{name}```和```{name}-neg```。
如果使用DreamArtist++方法，只在prompt中加入正触发词会有更好的效果。

### 加载插件

除了lora之外，还可以添加自定义插件。lora也可以当成自定义插件添加。比如加载`controlnet`。

首先转换官方模型为`hcp plugin`格式，例如转换`control_v11p_sd15_openpose`:
```yaml
python -m hcpdiff.tools.sd2diffusers --checkpoint_path "ckpts/control_v11p_sd15_openpose.pth" --original_config_file "ckpts/control_v11p_sd15_openpose.yaml" --dump_path "ckpts/control_v11p_sd15_openpose" --controlnet
```

在生成图片的配置文件中加入相关配置，并加载模型权重:
```yaml
merge:
  # 模型插件定义文件
  plugin_cfg: cfgs/plugins/plugin_controlnet.yaml
    
  group3:
    type: 'unet'
    base_model_alpha: 0.0 # 替换基础模型的层
    part: null
    lora: null
    plugin: # 加载模型插件
      controlnet1: # 插件名称，不能重复
        path: 'ckpts/controlnet.ckpt' # 模型路径
        layers: 'all' # 加载哪些层
```

## 单词注意力加强
可以单独加强部分词的注意力:
```
格式为: {加强的文本:倍率}，默认1.1倍
例如：
a {cat} running {in the {city}:1.2}
```
其中```cat```会被加强1.1倍，```in the```会被加强1.2倍，```city```则会被加强1.2*1.1倍。