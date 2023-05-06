# 公开模型使用教程

此教程在于介绍如何使用公开的模型，包括底层模型和lora模型，在本框架上进行推理

公开模型可以在[civitai](https://civitai.com/)上面下载

## 底层模型转换

直接下载的底层模型大多是.safetensors的单个文件格式，我们需要将他转换为diffuser文件格式. 如果模型结构不是diffusion v1.5，则需要自己指定--original_config_file

```python
python -m hcpdiff.tools.sd2diffusers.py --checkpoint_path your_model.safetensors --dump_path your_dump_path --from_safetensors
```

## lora模型转换

直接下载的lora模型大多是.safetensors的单个文件格式，我们需要对其权重名字进行转换以适应本框架. 转换后分为unet和text_encoder两个权重文件. 

```python
python -m hcpdiff.tools.sd2diffusers.py --lora_model_path your_lora.safetensors --dump_unet_path converted_unet_path.safetensors --dump_text_encdoer_path converted_text_encoder_path.safetensors 
```

## vae模型转换

本框架也支持加载vae模型，直接下载的vae模型可能是.safetensors的单个文件格式，同样需要进行转换

```python
python -m hcpdiff.tools.vae2diffusers --vae_pt_path your_vae.safetensors --dump_path your_dump_path --from_safetensors
```

## 例子

1. 在[civitai](https://civitai.com/)找到一个不错的底层模型[counterfeit-v30](https://civitai.com/models/4468/counterfeit-v30)，模型预览效果如下

<img src="../imgs/CounterfeitV30_sample.jpeg" style="zoom: 30%">

2. 下载后进行底层模型转换
3. 在[civitai](https://civitai.com/)找到一个akira的lora模型[akira-ray-v10](https://civitai.com/models/34147/akira-ray-nijisanji)
4. 下载后进行底层模型转换
5. 进行推理,注意底层模型，转换的lora模型的路径需要你进行替换

```python
python -m hcpdiff.visualizer --cfg cfgs/infer/public_model_infer.yaml
```

最终效果相当不错

<img src="../imgs/akira_ray_v10_output.png" style="zoom: 50%">