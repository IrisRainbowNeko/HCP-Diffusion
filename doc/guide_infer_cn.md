# 推理阶段使用教程

推理阶段同样使用```.yaml```配置文件描述使用哪些训练的组件。

## 加载训练的模型
可以指定多个模型进行分层融合，或者添加多个lora。

```yaml
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

## 单词注意力加强
可以单独加强部分词的注意力:
```
格式为: {加强的文本:倍率}，默认1.1倍
例如：
a {cat} running {in the {city}:1.2}
```
其中```cat```会被加强1.1倍，```in the```会被加强1.2倍，```city```则会被加强1.2*1.1倍。