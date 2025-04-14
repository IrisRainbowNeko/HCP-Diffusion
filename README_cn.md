# HCP-Diffusion V2

## 简介
HCP-Diffusion是一个基于[🐱 RainbowNeko Engine](https://github.com/IrisRainbowNeko/RainbowNekoEngine) 的Diffusion模型工具箱。
有清晰的代码结构，更方便的**python格式配置文件**，进行灵活的实验配置和管理。有丰富的训练组件支持。相比于现有框架更加灵活易用，可拓展性更强。

HCP-Diffusion可以通过一个```.py```配置文件用相似的配置方式，统一现有大多数训练方法和模型结构，包括Prompt-tuning (Textual Inversion), DreamArtist, Fine-tuning, DreamBooth, LoRA, ControlNet等绝大多数方法。
也可以实现各个方法直接的组合使用。

框架实现了基于LoRA的升级版DreamArtist，即DreamArtist++，只用一张图像就可以训练得到高泛化性，高可控性的LoRA。
相比DreamArtist更加稳定，图像质量和可控性更高，训练速度更快。

好主意！用表格确实可以让信息更紧凑、层次更清晰，尤其在 README 页面中，便于快速浏览和对比。下面是使用表格格式重新排版后的版本，保持专业和美观：

## 安装

安装 [pytorch](https://pytorch.org/)

通过pip安装:
```bash
pip install hcpdiff
# 初始化配置文件
hcpinit
```

从源码安装:
```bash
git clone https://github.com/7eu7d7/HCP-Diffusion.git
cd HCP-Diffusion
pip install -e .
# 初始化配置文件
hcpinit
```

使用xFormers减少显存使用并加速训练:
```bash
# 根据你的pytorch版本选择合适的xformers版本
pip install xformers==?
```

## 🚀 Python配置文件

RainbowNeko Engine支持由python语法习惯编写的配置文件，可以在配置中进行函数和类的调用，并且函数参数可以从父配置文件继承。框架会自动处理配置文件的格式。

比如下面的配置文件:
```python
dict(
    layer=Linear(in_features=4, out_features=4)
)
```
在解析阶段，会自动编译成:
```python
dict(
    layer=dict(_target_=Linear, in_features=4, out_features=4)
)
```
在解析之后，会由框架进行实例化。所以用户可以直接用python语法习惯编写配置文件。

---

## ✨ 特性支持

<details>
<summary>特性支持</summary>

### 📦 模型支持

| 模型名称      | 状态     |
|---------------|----------|
| Stable Diffusion 1.5 | ✅ 支持 |
| Stable Diffusion XL (SDXL) | ✅ 支持 |
| PixArt         | ✅ 支持 |
| FLUX           | 🚧 开发中 |
| Stable Diffusion 3 (SD3) | 🚧 开发中 |

---

### 🧠 微调功能

| 功能                     | 描述/支持情况 |
|--------------------------|----------------|
| LoRA 分层配置            | ✅ 支持（含 Conv2d） |
| 分层 Fine-Tuning         | ✅ 支持 |
| 多词联合 Prompt-Tuning   | ✅ 支持 |
| 分层模型融合             | ✅ 支持 |
| 自定义优化器             | ✅ 支持（Lion, DAdaptation, pytorch-optimizer 等） |
| 自定义学习率调度器       | ✅ 支持 |

---

### 🧩 拓展方法支持

| 方法                   | 状态     |
|------------------------|----------|
| ControlNet（含训练）    | ✅ 支持 |
| DreamArtist / DreamArtist++ | ✅ 支持 |
| Token 注意力调节        | ✅ 支持 |
| 最大句子长度拓展        | ✅ 支持 |
| Textual Inversion（自定义词） | ✅ 支持 |
| CLIP Skip              | ✅ 支持 |

---

### 🚀 训练加速支持

| 工具/库               | 支持模块       |
|------------------------|----------------|
| [🤗 Accelerate](https://github.com/huggingface/accelerate) | ✅ 支持 |
| [Colossal-AI](https://github.com/hpcaitech/ColossalAI) | ✅ 支持 |
| [xFormers](https://github.com/facebookresearch/xformers) | ✅ 支持 UNet 与文本编码器 |

---

### 🗂 数据集支持

| 功能                        | 描述/说明 |
|---------------------------|-----------|
| Aspect Ratio Bucket (ARB) | ✅ 自动聚类支持 |
| 多数据源/数据集支持                | ✅ 支持 |
| LMDB支持                    | ✅ 支持 |
| webdataset支持              | 🚧 开发中 |
| 图像局部注意力增强                 | ✅ 支持 |
| 标签打乱 & Dropout            | ✅ 多种标签编辑策略 |

---

### 📉 支持的 Loss 函数

| Loss 类型   | 描述         |
|-------------|--------------|
| Min-SNR     | ✅ 支持 |
| SSIM        | ✅ 支持 |
| GWLoss      | ✅ 支持 |

---

### 🌫 Diffusion 策略支持

| 策略类型     | 状态 |
|--------------|------|
| DDPM         | ✅ 支持 |
| EDM          | ✅ 支持 |
| Flow Matching| ✅ 支持 |

---

### 🧠 模型自动评估 (帮助挑选步数)

| 功能            | 描述/支持情况                             |
|---------------|-------------------------------------|
| 图像预览          | ✅ 支持 (可使用工作流预览)                     |
| FID           | 🚧 开发中                                |
| CLIP Score    | 🚧 开发中                                |
| CCIP Score    | 🚧 开发中                                |
| Corrupt Score | 🚧 开发中 |

---

### ⚡️ 图像生成

| 功能         | 描述/支持情况                            |
|------------|------------------------------------|
| 批量生成       | ✅ 支持                   |
| 批量prompt生成 | ✅ 支持                               |
| 图生图        | ✅ 支持                               |
| Inpaint    | ✅ 支持                               |
| Token注意力权重 | ✅ 支持 |

</details>

---

## 使用教程

### 训练

HCP-Diffusion提供了基于🤗 Accelerate的训练脚本。

```bash
# 多卡训练, 需要在cfgs/launcher/multi.yaml中配置使用的显卡
hcp_train --cfg cfgs/train/py/配置文件.py

# 单卡训练, 需要在cfgs/launcher/single.yaml中配置使用的显卡
hcp_train_1gpu --cfg cfgs/train/py/配置文件.py
```

配置文件中的参数，可以在命令中修改:
```bash
# 修改基础模型
hcp_train --cfg cfgs/train/py/配置文件.py model.wrapper.models.ckpt_path=pretrained_model_path
```

### 生成图像:
HCP-Diffusion使用python配置文件定义的工作流来生成图像:
```bash
hcp_run --cfg cfgs/workflow/text2img.py
```

同样也可以在命令中修改配置:
```bash
hcp_run --cfg cfgs/workflow/text2img_cli.py \
    pretrained_model=pretrained_model_path \
    prompt='positive_prompt' \
    negative_prompt='negative_prompt' \
    seed=42
```


### 📚 教程

+ 🧠 [模型训练教程](https://hcpdiff.readthedocs.io/zh-cn/latest/user_guides/train.html)
+ 🔧 [lora训练教程](https://hcpdiff.readthedocs.io/zh-cn/latest/tutorial/lora.html)
+ 🎨 [图像生成教程](https://hcpdiff.readthedocs.io/zh-cn/latest/user_guides/workflow.html)
+ ⚙️ [配置文件说明](https://hcpdiff.readthedocs.io/zh-cn/latest/user_guides/cfg.html)
+ 🧩 [模型格式说明](https://hcpdiff.readthedocs.io/zh-cn/latest/user_guides/model_format.html)

## 做出贡献

欢迎为工具箱贡献更多的模型与特性。

## 团队

该工具箱由 [中山大学HCP-Lab](https://www.sysu-hcp.net/) 维护。

## 引用

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
