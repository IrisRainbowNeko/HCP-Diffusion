# LoRA Training Tutorial

This guide will walk you through the process of training a LoRA model to add new knowledge to a Stable Diffusion (SD) base model.

The tutorial is divided into the following sections:

- Preparing the dataset
- Training the model
- Using LoRA to generate images
- Saving the model in SD-WebUI format

## Preparing the Dataset

Prepare the dataset you want to use for training. The dataset should consist of image-text pairs, where each image has a corresponding text annotation.

Place your training images in a folder, for example, in a folder named train_data, with the following structure:

```text
train_data
├── 1.png
├── 2.png
├── 3.png
├── ......
```

The framework supports multiple annotation formats and can automatically detect them. For more details, refer to the RainbowNeko Engine documentation: [Label File Formats](https://rainbownekoengine.readthedocs.io/zh-cn/stable/train/label_file.html)

::::{tab-set}
:::{tab-item} TXT Format

In the txt format, each annotation file shares the same name as the corresponding image, but with a .txt extension (e.g., 1.txt, 2.txt).

The text inside the .txt file should contain the annotation for the image.

You can place the text files in the same folder as the images:

```text
train_data
├── 1.png
├── 1.txt
├── 2.png
├── 2.txt
├── 3.png
├── 3.txt
├── ......
```

:::

:::{tab-item} JSON Format

In the json format, annotations for all images are stored in a single JSON file. The format is as follows:

```json
{
    "1": "annotation1",
    "2": "annotation2",
    "3": "annotation3",
    ......
}
```

:::

:::{tab-item} YAML Format

In the yaml format, annotations for all images are stored in a single YAML file. The format is as follows:

```yaml
"1": "annotation1"
"2": "annotation2"
"3": "annotation3"
......
```

:::
::::

## Lora Training

Once the dataset is ready, we can begin training the model.

It is recommended to install tensorboard or wandb to better monitor the training progress.

```bash
# Install tensorboard
pip install tensorboard

# Install wandb
pip install wandb
```

### Filling in the Training Configuration File

::::{tab-set}
:::{tab-item} Standard Configuration File

In cfgs/train/py/examples, there are template files provided for training. The file lora_preview.py is a template for training a LoRA model with preview functionality. You can inherit from this template to conduct training.

By default, the template only applies LoRA to the U-Net part, not the text encoder.

lora_preview.py inherits from SD_FT.py. The main configuration parameters to modify are as follows:

```python
...
model_plugin=CfgWDPluginParser(cfg_plugin=dict(
    lora1=LoraLayer.wrap_model(
        _partial_=True,
        lr=1e-4,  # LoRA learning rate
        rank=4,   # LoRA rank
        alpha=2,  # LoRA alpha, typically equal to or half of the rank
        layers=[
            're:denoiser.*\.attn.?',
            're:denoiser.*\.ff',
        ]
    )
), weight_decay=0.1),  # weight_decay is typically 0.1 or 0.01
...
train=dict(
    train_steps=1000,  # Total training steps
    # train_epochs=10, # Alternatively, use epochs instead of steps
    save_step=200,     # Save interval (in steps)

    # Optimizer: switch to 8-bit optimizer if VRAM is insufficient
    optimizer=torch.optim.AdamW(_partial_=True, betas=(0.9, 0.99)),

    # Learning rate scheduler: default is constant
    scheduler=ConstantLR(
        _partial_=True,
        warmup_steps=0,
    ),
),
...
model=dict(
    name='model',

    # ckpt_path is the path to the pretrained base model
    wrapper=SD15Wrapper.from_pretrained(
        models=SD15_auto_loader(ckpt_path='Lykon/DreamShaper', _partial_=True),
        _partial_=True,
    ),
),
...
evaluator=HCPPreviewer(_partial_=True,
    interval=100,  # Preview interval
    workflow=t2i_lora,  # Workflow used for image preview
),
```

During preview, the base model and LoRA plugins remain unchanged and use the model being trained.

Dataset configuration:

```python
@neko_cfg
def cfg_data():
    return dict(
        dataset1=TextImagePairDataset(_partial_=True,
            batch_size=4,  # Batch size
            loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root='imgs/',  # Image path
                    label_file='${.img_root}',  # Annotation path, defaults to same as image
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(
                bucket=RatioBucket,  # Use ARB
                word_names=dict(pt1='paimeng'),  # Trigger word
                erase=0,
            ),
            bucket=RatioBucket.from_files(
                target_area=512*512,  # Target training resolution
                num_bucket=4,  # Number of buckets (increase with more diverse resolutions)
            ),
            cache=VaeCache(bs=4)  # Cache VAE encodings to reduce VRAM usage
        )
    )
```

To save the LoRA model in both HCP-Diffusion and WebUI formats, configure ckpt_saver as follows:

```python
from rainbowneko.ckpt_manager import NekoPluginSaver, SafeTensorFormat
from hcpdiff.ckpt_manager import LoraWebuiFormat

ckpt_saver=dict(
    _replace_ = True,
    lora_unet=NekoPluginSaver(
        format=SafeTensorFormat(),
        target_plugin='lora1',
    ),
    lora_unet_webui=NekoPluginSaver(
        format=LoraWebuiFormat(),  # WebUI format
        target_plugin='lora1',
    ),
),
```

:::

:::{tab-item} Simplified Configuration File

In cfgs/train/py/easy, simplified training templates are provided. The file SD15_lora_preview.py is a LoRA training template with preview functionality.

By default, LoRA is only applied to the U-Net, not the text encoder.

```python
from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SD15_lora_train, cfg_data_SD_ARB, SD15_t2i
from hcpdiff.evaluate import HCPPreviewer

# Prompt used for preview
prompt = ('paimeng, 1girl, halo, white_hair, solo, smile, blue_eyes, looking_at_viewer, open_mouth, long_sleeves, white_dress, dress, single_thighhigh,'
          ' :d, cape, hair_between_eyes, thighhighs, hair_ornament, blush, white_outline, outline, sky, scarf, cloud, white_thighhighs, arm_up,'
          ' notice_lines, paimon_(genshin_impact)')

@neko_cfg
def make_cfg():
    return dict(
        **SD15_lora_train(
            base_model='Lykon/DreamShaper',  # Path to pretrained model
            train_steps=1000,  # Total training steps
            save_step=200,  # Save interval
            rank=8,  # LoRA rank
            dataset=dict(
                dataset1=cfg_data_SD_ARB(  # Use ARB to avoid cropping
                    img_root='imgs/',  # Path to training images
                    batch_size=4,  # Batch size
                    trigger_word='paimeng',  # Trigger word
                    resolution=512*512,  # Training resolution
                    num_bucket=4,  # Number of buckets
                )
            )
        ),
        # Enable preview
        evaluator=HCPPreviewer(_partial_=True,
            interval=100,  # Preview interval
            workflow=SD15_t2i(
                pretrained_model='${model.wrapper.models.ckpt_path}',
                prompt=prompt,
                seed=42,  # Fixed random seed
            ),
        ),
    )
```

Additional optional configurations:

```python
@neko_cfg
def make_cfg():
    return dict(
        **SD15_lora_train(
            base_model='Lykon/DreamShaper',
            train_steps=1000,
            save_step=200,
            rank=8,

            lr=1e-4,  # Learning rate
            alpha=8,  # LoRA alpha
            clip_skip=0,  # Number of skipped CLIP layers (0 = none)
            with_conv=False,  # Whether to apply LoRA to convolution layers
            low_vram=False,  # Use 8-bit optimizer
            save_webui_format=True,  # Save in WebUI format

            dataset=dict(
                dataset1=cfg_data_SD_ARB(
                    img_root='imgs/',
                    batch_size=4,
                    trigger_word='paimeng',
                    resolution=512*512,
                    num_bucket=4,
                )
            )
        ),
        evaluator=HCPPreviewer(_partial_=True,
            interval=100,
            workflow=SD15_t2i(
                pretrained_model='${model.wrapper.models.ckpt_path}',
                prompt=prompt,
                seed=42,

                negative_prompt='',  # Negative prompt
                noise_sampler=Diffusers_SD.dpmpp_2m_karras,  # Sampler
                bs=4,
                width=512,
                height=512,
                N_steps=20,  # Sampling steps
                guidance_scale=7.0,  # CFG guidance scale
            ),
        ),
    )
```

:::
::::

### Training

::::{tab-set}
:::{tab-item} Single GPU Training

```bash
hcp_train_1gpu --cfg cfgs/train/py/examples/lora_preview.py
```

You can override config values via CLI:

```bash
hcp_train_1gpu --cfg cfgs/train/py/examples/lora_preview.py train.train_epochs=10
```
:::

:::{tab-item} Multi-GPU Training

Specify the GPU IDs and number of GPUs in cfgs/launcher/multi.yaml, then run:

```bash
hcp_train --cfg cfgs/train/py/examples/lora_preview.py
```

Override config values via CLI:

```bash
hcp_train --cfg cfgs/train/py/examples/lora_preview.py train.train_epochs=10
```
:::
::::

After training, all outputs are saved in a folder:

```text
exps/2023-07-26-01-05-35
├── cfg.py         # Configuration file
├── ckpts          # LoRA model checkpoints
│   ├── lora_unet-100.safetensors
│   ├── lora_unet-200.safetensors
│   ├── ...
├── tblog          # Tensorboard logs
│   └── events.out.tfevents.1690346085.myenvironment.210494.0
└── train.log      # Training log
```

### Advanced Configuration

Customize LoRA parameters and target layers:

```python
lora_unet=LoraLayer.wrap_model(
    _partial_=True,
    lr=1e-4,
    rank=4,
    alpha=2,
    layers=[
        're:.*\.to_k',
        're:.*\.to_v',
        're:.*\.ff',
    ]
),
lora_TE=LoraLayer.wrap_model(
    _partial_=True,
    lr=2e-5,
    rank=2,
    alpha=2,
    layers=[
        're:.*self_attn',
        're:.*mlp',
    ]
)
```

## Using LoRA to Generate Images

After training, you can use the LoRA model to generate images. For detailed instructions, refer to [Workflow Configuration Guide](../user_guides/workflow.md).

You can use the simplified config file: cfgs/workflow/easy/t2i_lora.py

```python
from hcpdiff.easy.cfg import SD15_t2i_lora
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SD15_t2i_lora(
        pretrained_model='Lykon/DreamShaper',
        lora_info=[
            ('exps/2023-07-26-01-05-35/ckpts/lora_unet-1000.safetensors', 1.0),
        ],
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        bs=4,
        width=512,
        height=512,
        guidance_scale=7.0
    )
```

Run the config to generate images:

```shell
hcp_run --cfg cfgs/workflow/easy/t2i_lora.py
```

### Advanced Settings

You can also change the sampler and VAE in the configuration:

```python
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from hcpdiff.diffusion.sampler import DiffusersSampler

wrapper=SD15Wrapper.from_pretrained(
    _partial_=True,
    models=SD15_auto_loader(
        ckpt_path=base_model,
        vae=AutoencoderKL.from_pretrained('any3/vae'),  # Replace VAE
        noise_sampler=DiffusersSampler(  # Replace sampler
            DPMSolverMultistepScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule='scaled_linear',
                algorithm_type='sde-dpmsolver++',
                use_karras_sigmas=True,
            )
        ),
        _partial_=True
    ),
),
```