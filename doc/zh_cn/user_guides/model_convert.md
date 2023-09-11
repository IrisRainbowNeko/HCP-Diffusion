# 与webui格式模型转换

此教程在于介绍如何将webui格式的公开模型与本框架模型相互转换，包括底层模型和lora模型。

## 基础模型转换

直接下载的基础模型大多是```.ckpt```或```.safetensors```的单个文件格式，我们需要将他转换为diffuser文件格式。

+ 首先下载 [配置文件](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-inference.yaml)
+ 根据配置文件转换模型

```bash
python -m hcpdiff.tools.sd2diffusers \
    --checkpoint_path "stable diffusion模型路径" \
    --original_config_file "下载的配置文件路径" \
    --dump_path "储存路径(文件夹)" 
    [--extract_ema] # 是否提取ema模型
    [--from_safetensors] # 原模型是safetensors格式添加
    [--to_safetensors] # 是否存成safetensors格式
```

## lora模型转换

### 从webui格式转换
webui使用的lora模型不能直接被加载，需要对其权重名字进行转换以适应本框架。转换后分为```unet```和```text_encoder```两个权重文件:

```bash
python -m hcpdiff.tools.lora_convert --from_webui --lora_path lora.safetensors --dump_path lora_hcp/ \
      --auto_scale_alpha # 现有webui模型没有alpha自动缩放，需要转换
```

### 转换为webui格式
也可以将本框架训练得到的lora模型转换成webui支持的格式:

```bash
python -m hcpdiff.tools.lora_convert --to_webui --lora_path unet-lora.safetensors --lora_path_TE text_encoder-lora.safetensors --dump_path lora-webui.safetensors \
      --auto_scale_alpha # 现有webui模型没有alpha自动缩放，需要转换
```

## vae模型转换

本框架也支持单独加载vae模型，直接下载的vae模型是```.pt```或```.safetensors```的单个文件格式，同样需要进行转换:

```bash
python -m hcpdiff.tools.sd2diffusers \
    --vae_pt_path "vae模型路径" \
    --original_config_file "下载的配置文件路径" \
    --dump_path "储存路径(文件夹)" 
    [--from_safetensors] # 原模型是safetensors格式添加
```

## 例子

1. 在 [civitai](https://civitai.com/) 上下基础模型 [counterfeit-v30](https://civitai.com/models/4468/counterfeit-v30) ，模型预览效果如下:

<img src="../imgs/CounterfeitV30_sample.jpeg" style="zoom: 30%">

2. 下载后进行基础模型转换
3. 在 [civitai](https://civitai.com/) 下载一个lora模型 [akira-ray-v10](https://civitai.com/models/34147/akira-ray-nijisanji)
4. 下载后进行lora模型转换
5. 进行推理。注意底层模型，转换的lora模型的路径需要你进行替换

```bash
python -m hcpdiff.visualizer --cfg cfgs/infer/webui_model_infer.yaml
```

使用webui转换的lora模型，需要在lora参数中添加```alpha_auto_scale: False```:
```yaml
merge: 
  group1:
    type: 'unet'
    base_model_alpha: 1.0
    lora:
      - path: 'unet-100.safetensors'
        alpha: 0.6
        layers: 'all'
        alpha_auto_scale: False # 关闭alpha自动缩放
    part: null
```

最终效果:

<img src="../imgs/akira_ray_v10_output.png" style="zoom: 50%">