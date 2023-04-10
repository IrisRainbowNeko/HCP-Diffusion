# HCP-Diffusion

[📘中文说明](./README_cn.md)

## Introduction
HCP-Diffusion is a toolbox for stable diffusion models based on diffusers.
It facilitates flexiable configurations and component support for training, in comparison with webui and sd-scripts.

This toolbox supports **colossal-AI**, which can significantly reduce GPU memory usage.

HCP-Diffusion can unify existing training methods for text-to-image generation (e.g., Prompt-tuning, Textual Inversion, DreamArtist, Fine-tuning, DreamBooth, LoRA, ControlNet, etc) and model structures through a single ```.yaml``` configuration file.

The toolbox has also implemented an upgraded version of DreamArtist with LoRA, named DreamArtist++, for one-shot text-to-image generation.
Compared to DreamArtist, DreamArtist++ is more stable with higher image quality and generation controllability, and faster training speed.

## Features

* Layer-wise LoRA (with Conv2d)
* Layer-wise fine-tuning
* Layer-wise model ensemble
* Prompt-tuning with multiple words
* DreamArtist and DreamArtist++
* Aspect Ratio Bucket (ARB) with automatic clustering
* Priori data
* Image attention mask
* Word attention multiplier
* Custom words that occupy multiple words
* Maximum sentence length expansion
* colossal-AI
* xformers for unet and text-encoder
* CLIP skip
* Tag shuffle and dropout
* safetensors support

## Install
```bash
git clone https://github.com/7eu7d7/HCP-Diffusion.git
cd HCP-Diffusion
pip install -r requirements.txt
```

## User guidance

Training:
```yaml
# with accelerate
accelerate launch train_ac.py --cfg cfgs/train/cfg_file.yaml
# with accelerate and only one gpu
accelerate launch train_ac_single.py --cfg cfgs/train/cfg_file.yaml
# with colossal-AI
torchrun --nproc_per_node 1 train_colo.py --cfg cfgs/train/cfg_file.yaml
```

Inference:
```yaml
python visualizer.py --pretrained_model pretrained_model_path
        --prompt positive_prompt \
        --neg_prompt negative_prompt \
        --seed 42 \
        [--cfg_merge cfg_file_of_load_lora_or_model_part]
```

+ [Model Training Tutorial](doc/guide_train.md)
+ [DreamArtist++ Tutorial](doc/guide_DA_cn.md)
+ [Model Inference Tutorial](doc/guide_infer_cn.md)

## Team

This toolbox is maintained by [HCP-Lab, SYSU](https://www.sysu-hcp.net/).
More models and features are welcome to contribute to this toolbox.

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