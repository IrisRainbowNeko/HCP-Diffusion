# HCP-Diffusion

[![PyPI](https://img.shields.io/pypi/v/hcpdiff)](https://pypi.org/project/hcpdiff/)
[![GitHub stars](https://img.shields.io/github/stars/7eu7d7/HCP-Diffusion)](https://github.com/7eu7d7/HCP-Diffusion/stargazers)
[![GitHub license](https://img.shields.io/github/license/7eu7d7/HCP-Diffusion)](https://github.com/7eu7d7/HCP-Diffusion/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/7eu7d7/HCP-Diffusion/branch/main/graph/badge.svg)](https://codecov.io/gh/7eu7d7/HCP-Diffusion)
[![open issues](https://isitmaintained.com/badge/open/7eu7d7/HCP-Diffusion.svg)](https://github.com/7eu7d7/HCP-Diffusion/issues)

[📘中文说明](./README_cn.md)

## Introduction
HCP-Diffusion is a toolbox for Stable Diffusion models based on [🤗 Diffusers](https://github.com/huggingface/diffusers).
It facilitates flexiable configurations and component support for training, in comparison with [webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [sd-scripts](https://github.com/kohya-ss/sd-scripts).

This toolbox supports [**Colossal-AI**](https://github.com/hpcaitech/ColossalAI), which can significantly reduce GPU memory usage.

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
* Multiple datasets with multiple data sources
* Image attention mask
* Word attention multiplier
* Custom words that occupy multiple words
* Maximum sentence length expansion
* [🤗 Accelerate](https://github.com/huggingface/accelerate)
* [Colossal-AI](https://github.com/hpcaitech/ColossalAI)
* [xFormers](https://github.com/facebookresearch/xformers) for UNet and text-encoder
* CLIP skip
* Tag shuffle and dropout
* [Safetensors](https://github.com/huggingface/safetensors) support
* [Controlnet](https://github.com/lllyasviel/ControlNet) (support training)
* Min-SNR loss
* Custom optimizer (Lion, DAdaptation, pytorch-optimizer, ...)
* Custom lr scheduler
* SDXL support

## Install

Install with pip:
```bash
pip install hcpdiff
# Start a new project and make initialization
hcpinit
```

Install from source:
```bash
git clone https://github.com/7eu7d7/HCP-Diffusion.git
cd HCP-Diffusion
pip install -e .
# Modified based on this project or start a new project and make initialization
## hcpinit
```

To use xFormers to reduce VRAM usage and accelerate training:
```bash
# use conda
conda install xformers -c xformers

# use pip
pip install xformers>=0.0.17
```

## User guidance

### Training

Training scripts based on 🤗 Accelerate or Colossal-AI are provided.
+ For 🤗 Accelerate, you may need to [configure the environment](https://github.com/huggingface/accelerate/tree/main#launching-script) before launching the scripts.
+ For Colossal-AI, you can use [torchrun](https://pytorch.org/docs/stable/elastic/run.html) to launch the scripts.

```yaml
# with Accelerate
accelerate launch -m hcpdiff.train_ac --cfg cfgs/train/cfg_file.yaml
# with Accelerate and only one GPU
accelerate launch -m hcpdiff.train_ac_single --cfg cfgs/train/cfg_file.yaml
# with Colossal-AI
torchrun --nproc_per_node 1 -m hcpdiff.train_colo --cfg cfgs/train/cfg_file.yaml
```

### Inference
```yaml
python -m hcpdiff.visualizer --cfg cfgs/infer/cfg.yaml pretrained_model=pretrained_model_path \
        prompt='positive_prompt' \
        neg_prompt='negative_prompt' \
        seed=42
```

### Conversion of Stable Diffusion models
The framework is based on 🤗 Diffusers. So it needs to convert the original Stable Diffusion model into a supported format using the [scripts provided by 🤗 Diffusers](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py).
+ Download the [config file](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-inference.yaml)
+ Convert models based on config file

```bash
python -m hcpdiff.tools.sd2diffusers \
    --checkpoint_path "path_to_stable_diffusion_model" \
    --original_config_file "path_to_config_file" \
    --dump_path "save_directory" \
    [--extract_ema] # Extract EMA model
    [--from_safetensors] # Whether the original model is in safetensors format
    [--to_safetensors] # Whether to save to safetensors format
```

Convert VAE:
```bash
python -m hcpdiff.tools.sd2diffusers \
    --vae_pt_path "path_to_VAE_model" \
    --original_config_file "path_to_config_file" \
    --dump_path "save_directory"
    [--from_safetensors]
```

### Tutorials
+ [Model Training Tutorial](doc/guide_train.md)
+ [DreamArtist++ Tutorial](doc/guide_DA.md)
+ [Model Inference Tutorial](doc/guide_infer.md)
+ [Configuration File Explanation](doc/guide_cfg.md)
+ [webui Model Conversion Tutorial](doc/guide_webui_lora.md)

## Contributing

You are welcome to contribute more models and features to this toolbox!

## Team

This toolbox is maintained by [HCP-Lab, SYSU](https://www.sysu-hcp.net/).

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
