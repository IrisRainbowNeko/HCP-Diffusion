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
* 先验数据集 (正则化)
* Image attention mask
* 单词注意力调整
* 占多个词位置的自定义单词
* 最大句子长度拓展
* colossal-AI
* xformers for unet and text-encoder
* CLIP skip
* Tag shuffle and dropout
* safetensors支持

## 安装
```bash
git clone https://github.com/7eu7d7/HCP-Diffusion.git
cd HCP-Diffusion
pip install -r requirements.txt
```

## 使用教程

训练命令:
```yaml
# with accelerate
accelerate launch train_ac.py --cfg cfgs/train/配置文件.yaml
# with accelerate and only one gpu
accelerate launch train_ac_single.py --cfg cfgs/train/配置文件.yaml
# with colossal-AI
torchrun --nproc_per_node 1 train_colo.py --cfg cfgs/train/配置文件.yaml
```

生成图像:
```yaml
python visualizer.py --pretrained_model pretrained_model_path
        --prompt positive_prompt \
        --neg_prompt negative_prompt \
        --seed 42 \
        [--cfg_merge cfg_file_of_load_lora_or_model_part]
```

+ [模型训练教程](doc/guide_train_cn.md)
+ [DreamArtist++使用教程](doc/guide_DA_cn.md)
+ [图像生成教程](doc/guide_infer_cn.md)
+ [配置文件说明](doc/guide_cfg_cn.md)

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