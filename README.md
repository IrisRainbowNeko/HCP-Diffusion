# HCP-Diffusion V2

[![PyPI](https://img.shields.io/pypi/v/hcpdiff)](https://pypi.org/project/hcpdiff/)
[![GitHub stars](https://img.shields.io/github/stars/7eu7d7/HCP-Diffusion)](https://github.com/7eu7d7/HCP-Diffusion/stargazers)
[![GitHub license](https://img.shields.io/github/license/7eu7d7/HCP-Diffusion)](https://github.com/7eu7d7/HCP-Diffusion/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/7eu7d7/HCP-Diffusion/branch/main/graph/badge.svg)](https://codecov.io/gh/7eu7d7/HCP-Diffusion)
[![open issues](https://isitmaintained.com/badge/open/7eu7d7/HCP-Diffusion.svg)](https://github.com/7eu7d7/HCP-Diffusion/issues)

[📘中文说明](./README_cn.md)

[📘English document](https://hcpdiff.readthedocs.io/en/latest/)
[📘中文文档](https://hcpdiff.readthedocs.io/zh_CN/latest/)

## Introduction

**HCP-Diffusion** is a Diffusion model toolbox built on top of the [🐱 RainbowNeko Engine](https://github.com/IrisRainbowNeko/RainbowNekoEngine).  
It features a clean code structure and a flexible **Python-based configuration file**, making it easier to conduct and manage complex experiments. It includes a wide variety of training components, and compared to existing frameworks, it's more extensible, flexible, and user-friendly.

HCP-Diffusion allows you to use a single `.py` config file to unify training workflows across popular methods and model architectures, including Prompt-tuning (Textual Inversion), DreamArtist, Fine-tuning, DreamBooth, LoRA, ControlNet, ....  
Different techniques can also be freely combined.

This framework also implements **DreamArtist++**, an upgraded version of DreamArtist based on LoRA. It enables high generalization and controllability with just a single image for training.  
Compared to the original DreamArtist, it offers better stability, image quality, controllability, and faster training.

---

## Installation

Install via pip:

```bash
pip install hcpdiff
# Initialize configuration
hcpinit
```

Install from source:

```bash
git clone https://github.com/7eu7d7/HCP-Diffusion.git
cd HCP-Diffusion
pip install -e .
# Initialize configuration
hcpinit
```

Use xFormers to reduce memory usage and accelerate training:

```bash
# Choose the appropriate xformers version for your PyTorch version
pip install xformers==?
```

## 🚀 Python Configuration Files
RainbowNeko Engine supports configuration files written in a Python-like syntax. This allows users to call functions and classes directly within the configuration file, with function parameters inheritable from parent configuration files. The framework automatically handles the formatting of these configuration files.

For example, consider the following configuration file:
```python
dict(
    layer=Linear(in_features=4, out_features=4)
)
```
During parsing, this will be automatically compiled into:
```python
dict(
    layer=dict(_target_=Linear, in_features=4, out_features=4)
)
```
After parsing, the framework will instantiate the components accordingly. This means users can write configuration files using familiar Python syntax.

---

## ✨ Features

<details>
<summary>Features</summary>

### 📦 Model Support

| Model Name                | Status      |
|--------------------------|-------------|
| Stable Diffusion 1.5     | ✅ Supported |
| Stable Diffusion XL (SDXL)| ✅ Supported |
| PixArt                   | ✅ Supported |
| FLUX                     | 🚧 In Development |
| Stable Diffusion 3 (SD3) | 🚧 In Development |

---

### 🧠 Fine-Tuning Capabilities

| Feature                         | Description/Support |
|----------------------------------|---------------------|
| LoRA Layer-wise Configuration   | ✅ Supported (including Conv2d) |
| Layer-wise Fine-Tuning          | ✅ Supported |
| Multi-token Prompt-Tuning       | ✅ Supported |
| Layer-wise Model Merging        | ✅ Supported |
| Custom Optimizers               | ✅ Supported (Lion, DAdaptation, pytorch-optimizer, etc.) |
| Custom LR Schedulers            | ✅ Supported |

---

### 🧩 Extension Method Support

| Method                         | Status      |
|--------------------------------|-------------|
| ControlNet (including training)| ✅ Supported |
| DreamArtist / DreamArtist++    | ✅ Supported |
| Token Attention Adjustment     | ✅ Supported |
| Max Sentence Length Extension  | ✅ Supported |
| Textual Inversion (Custom Tokens)| ✅ Supported |
| CLIP Skip                      | ✅ Supported |

---

### 🚀 Training Acceleration

| Tool/Library                                       | Supported Modules        |
|---------------------------------------------------|---------------------------|
| [🤗 Accelerate](https://github.com/huggingface/accelerate)    | ✅ Supported |
| [Colossal-AI](https://github.com/hpcaitech/ColossalAI)       | ✅ Supported |
| [xFormers](https://github.com/facebookresearch/xformers)     | ✅ Supported (UNet and text encoder) |

---

### 🗂 Dataset Support

| Feature                         | Description |
|----------------------------------|-------------|
| Aspect Ratio Bucket (ARB)       | ✅ Auto-clustering supported |
| Multi-source / Multi-dataset    | ✅ Supported |
| LMDB                            | ✅ Supported |
| webdataset                      | 🚧 In Development |
| Local Attention Enhancement     | ✅ Supported |
| Tag Shuffling & Dropout         | ✅ Multiple tag editing strategies |

---

### 📉 Supported Loss Functions

| Loss Type  | Description |
|------------|-------------|
| Min-SNR    | ✅ Supported |
| SSIM       | ✅ Supported |
| GWLoss     | ✅ Supported |

---

### 🌫 Supported Diffusion Strategies

| Strategy Type   | Status       |
|------------------|--------------|
| DDPM             | ✅ Supported |
| EDM              | ✅ Supported |
| Flow Matching    | ✅ Supported |

---

### 🧠 Automatic Evaluation (Step Selection Assistant)

| Feature         | Description/Status                       |
|------------------|------------------------------------------|
| Image Preview    | ✅ Supported (workflow preview)           |
| FID              | 🚧 In Development                        |
| CLIP Score       | 🚧 In Development                        |
| CCIP Score       | 🚧 In Development                        |
| Corrupt Score    | 🚧 In Development                        |

</details>

---

## Getting Started

### Training

HCP-Diffusion provides training scripts based on 🤗 Accelerate.

```bash
# Multi-GPU training, configure GPUs in cfgs/launcher/multi.yaml
hcp_train --cfg cfgs/train/py/your_config.py

# Single-GPU training, configure GPU in cfgs/launcher/single.yaml
hcp_train_1gpu --cfg cfgs/train/py/your_config.py
```

You can also override config items via command line:

```bash
# Override base model path
hcp_train --cfg cfgs/train/py/your_config.py model.wrapper.models.ckpt_path=pretrained_model_path
```

### Image Generation

Use the workflow defined in the Python config to generate images:

```bash
hcp_run --cfg cfgs/workflow/text2img.py
```

Or override parameters via command line:

```bash
hcp_run --cfg cfgs/workflow/text2img_cli.py \
    pretrained_model=pretrained_model_path \
    prompt='positive_prompt' \
    negative_prompt='negative_prompt' \
    seed=42
```

### Tutorials

🚧 In Development

---

## Contributing

We welcome contributions to support more models and features.

---

## Team

Maintained by [HCP-Lab at Sun Yat-sen University](https://www.sysu-hcp.net/).

---

## Citation

```bibtex
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
