# HCP-Diffusion

## 简介
HCP-Diffusion是一个基于diffusers的stable diffusion模型训练工具箱。
相比于webui和sd-scripts有更加清晰的代码结构，更方便的实验配置和管理方法，更多的训练组件支持。

框架支持**colossal-AI**，可以大幅减少显存消耗。

HCP-Diffusion可以通过一个```.yaml```配置文件统一现有大多数训练方法和模型结构，包括，
Prompt-tuning (textual inversion), DreamArtist, Fine-tuning, DreamBooth, lora, controlnet等绝大多数方法。
也可以实现各个方法直接的组合使用。

框架实现了基于lora升级版的DreamArtist++，只用一张图像就可以训练得到高泛化性，高可控性的lora。
相比DreamArtist更加稳定，图像质量和可控性更高，训练速度更快。

## 特性支持

* 分层添加lora (包含Conv2d层)
* 分层fine-tuning
* 分层模型融合
* 多个单词联合Prompt-tuning
* DreamArtist and DreamArtist++
* 带自动聚类的Aspect Ratio Bucket (ARB)
* 支持多个数据源的多个数据集
* 图像局部注意力强化
* 单词注意力调整
* 占多个词位置的自定义单词
* 最大句子长度拓展
* colossal-AI
* xformers for unet and text-encoder
* CLIP skip
* 标签打乱和 dropout
* safetensors支持
* Controlnet (支持训练)
* Min-SNR loss
* 自定义优化器 (Lion, DAdaptation, pytorch-optimizer, ...)
* 自定义学习率调整器

## 安装
通过pip安装:
```bash
pip install hcpdiff
# 新建一个项目并进行初始化
hcpinit
```

从源码安装:
```bash
git clone https://github.com/7eu7d7/HCP-Diffusion.git
cd HCP-Diffusion
pip install -e .
# 基于此项目直接修改，或新建一个项目并进行初始化
## hcpinit
```

## 使用教程

训练命令:
```yaml
# with accelerate
accelerate launch -m hcpdiff.train_ac --cfg cfgs/train/配置文件.yaml
# with accelerate and only one gpu
accelerate launch -m hcpdiff.train_ac_single --cfg cfgs/train/配置文件.yaml
# with colossal-AI
torchrun --nproc_per_node 1 -m hcpdiff.train_colo --cfg cfgs/train/配置文件.yaml
```

生成图像:
```yaml
python -m hcpdiff.visualizer --cfg cfgs/infer/cfg.yaml pretrained_model=pretrained_model_path \
        prompt='positive_prompt' \
        neg_prompt='negative_prompt' \
        seed=42
```

该框架基于diffusers，可以使用 [diffusers提供的脚本](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py)
把原版stable diffusion模型转换成支持的格式:
+ 首先下载[配置文件](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-inference.yaml)
+ 根据配置文件转换模型

```bash
python -m hcpdiff.tools.sd2diffusers \
    --checkpoint_path "stable diffusion模型路径" \
    --original_config_file "下载的配置文件路径" \
    --dump_path "储存路径(文件夹)" \
    [--extract_ema] # 是否提取ema模型
    [--from_safetensors] # 原模型是不是safetensors格式
    [--to_safetensors] # 是否存成safetensors格式
```

转换VAE:
```bash
python -m hcpdiff.tools.sd2diffusers \
    --vae_pt_path "VAE模型路径" \
    --original_config_file "下载的配置文件路径" \
    --dump_path "储存路径(文件夹)"
    [--from_safetensors]
```

+ [模型训练教程](doc/guide_train_cn.md)
+ [DreamArtist++使用教程](doc/guide_DA_cn.md)
+ [图像生成教程](doc/guide_infer_cn.md)
+ [配置文件说明](doc/guide_cfg_cn.md)
+ [webui模型转换教程](doc/guide_webui_lora_cn.md)

使用xformers减少显存使用并加速训练:
```bash
# use conda
conda install xformers -c xformers

# use pip
pip install xformers>=0.0.17
```

## Team

该工具箱由 [HCP-Lab, SYSU](https://www.sysu-hcp.net/) 维护，
欢迎为工具箱贡献更多的模型与特性。

## Citation

```
@article{DBLP:journals/corr/abs-2211-11337,
  author    = {Ziyi Dong and
               Pengxu Wei and
               Liang Lin},
  title     = {DreamArtist: Towards Controllable One-Shot Text-to-Image Generation
               via Positive-Negative Prompt-Tuning},
  journal   = {CoRR},
  volume    = {abs/2211.11337},
  year      = {2022},
  doi       = {10.48550/arXiv.2211.11337},
  eprinttype = {arXiv},
  eprint    = {2211.11337},
}
```
