# Model Fine-Tuning Tutorial

This guide will walk you through how to fine-tune a model. Starting from a pre-trained base model, you will continue training using your own dataset.

This tutorial is divided into the following sections:

- Preparing the dataset
- Training the model
- Generating images
- Saving in sd-webui format

## Preparing the Dataset

Prepare the dataset you want to use for training. The dataset should contain image-caption pairs, where each image corresponds to a specific text description.

Place the training images in a folder, for example in a folder named train_data, with the structure as follows:

```text
train_data
├── 1.png
├── 2.png
├── 3.png
├── ......
```

The text annotations support multiple formats, and the framework can automatically recognize them. For more details, refer to the RainbowNeko Engine documentation:
[RainbowNeko Engine - Label File](https://rainbownekoengine.readthedocs.io/zh-cn/stable/train/label_file.html)

::::{tab-set}
:::{tab-item} txt Format

In the txt format, the annotation file has the same name as the corresponding image file, but with a .txt extension (e.g., 1.txt, 2.txt, etc.). The annotation content should be written inside the txt file. These files can be placed in the same folder as the images.

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

:::{tab-item} json Format

In json format, all annotations are stored in a single JSON file with the following structure:

```json
{
    "1": "Annotation 1",
    "2": "Annotation 2",
    "3": "Annotation 3",
    ......
}
```

:::

:::{tab-item} yaml Format

In yaml format, all annotations are stored in a single YAML file with the following structure:

```yaml
"1": "Annotation 1"
"2": "Annotation 2"
"3": "Annotation 3"
......
```

:::
::::

## Model Training

Once the dataset is ready, you can start training the model.

It is recommended to install tensorboard or wandb to better monitor the training progress.

```bash
# Install tensorboard
pip install tensorboard

# Install wandb
pip install wandb
```

### Filling in the Training Configuration File

:::::{tab-set}
::::{tab-item} Standard Config File

Under cfgs/train/py/examples, there are template files for training. The SD_FT.py file provides a template for training the SD1.5 model. You can copy this file to start training.

By default, the template only fine-tunes the U-Net component.

Copy the configuration file to cfgs/train/py/, for example as ft1.py, and modify the following key settings:

```python
from cfgs.workflow import text2img

model_part = CfgWDModelParser([
    dict(
        lr=1e-5,  # Learning rate
        layers=['denoiser'],  # Specify layers to train, here it's U-Net
    )
], weight_decay=1e-2),  # Weight decay, usually 0.01 or 0.001
...
train = dict(
    train_steps=1000,  # Total training steps
    # train_epochs=10, # Alternatively, specify number of epochs
    save_step=200,  # Save model every N steps

    # Optimizer; switch to 8-bit optimizer if VRAM is limited
    optimizer=torch.optim.AdamW(_partial_=True, betas=(0.9, 0.99)),

    # Learning rate scheduler, default is constant
    scheduler=ConstantLR(
        _partial_=True,
        warmup_steps=0,
    ),
),
...
model = dict(
    name='model',

    # ckpt_path is the path to the pre-trained base model
    wrapper=SD15Wrapper.from_pretrained(
        models=SD15_auto_loader(ckpt_path='Lykon/DreamShaper', _partial_=True),
        _partial_=True,
    ),
),
...
# Add preview functionality
evaluator = HCPPreviewer(_partial_=True,
    interval=100,  # Preview interval
    workflow=text2img,  # Workflow used to preview images
),
```

Fill in the parameters and paths in the config file according to your actual setup. The preview will use the model being trained.

Dataset configuration example:

```python
@neko_cfg
def cfg_data():
    return dict(
        dataset1=TextImagePairDataset(_partial_=True,
            batch_size=4,  # Batch size
            loss_weight=1.0,  # Weight of this dataset
            source=dict(
                data_source1=Text2ImageSource(
                    img_root='imgs/',  # Path to images
                    label_file='${.img_root}',  # Path to annotations, same as image folder
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(bucket=RatioBucket),  # Same bucket type as below
            bucket=RatioBucket.from_files(
                target_area=512*512,  # Training resolution
                num_bucket=6,  # More buckets = more resolution variations
            ),
            cache=VaeCache(bs=4)  # Cache VAE encodings to save VRAM
        )
    )
```
::::
::::{tab-item} Simplified Config File

Under cfgs/train/py/easy, simplified template files are available. The SD15_FT.py file provides a template for fine-tuning SD1.5. You can use this as a base for training.

By default, only the U-Net part is trained.

```python
from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SD15_finetuning, cfg_data_SD_ARB

@neko_cfg
def make_cfg():
    return SD15_finetuning(
        base_model='Lykon/DreamShaper',  # Path to pre-trained model
        train_steps=1000,  # Total training steps
        save_step=200,  # Save model every N steps
        dataset=dict(
            dataset1=cfg_data_SD_ARB(  # Use ARB to avoid cropping images
                img_root='imgs/',  # Path to training images
                batch_size=4,  # Batch size
                resolution=512*512,  # Training resolution
                num_bucket=4,  # More buckets = more resolution variations
            )
        )
    )
```

:::{dropdown} Add Preview Functionality
:animate: fade-in

Add simple configuration to enable preview functionality for the model:

```python
from rainbowneko.parser import neko_cfg
from hcpdiff.easy.cfg import SD15_finetuning, cfg_data_SD_ARB, SD15_t2i
from hcpdiff.evaluate import HCPPreviewer

@neko_cfg
def make_cfg():
    return dict(
        **SD15_finetuning(
            base_model='Lykon/DreamShaper',
            train_steps=1000,
            save_step=200,
            dataset=dict(
                dataset1=cfg_data_SD_ARB(
                    img_root='imgs/',
                    batch_size=4,
                    resolution=512*512,
                    num_bucket=4,
                )
            )
        ),
        evaluator=HCPPreviewer(_partial_=True,
            interval=100,
            workflow=SD15_t2i(
                pretrained_model='${model.wrapper.models.ckpt_path}',
                prompt='',  # Prompt for preview
                seed=42,  # Set random seed

                ## Optional settings
                negative_prompt='',  # Negative prompt
                noise_sampler=Diffusers_SD.dpmpp_2m_karras,  # Sampler
                bs=4,  # Batch size
                width=512,
                height=512,
                N_steps=20,  # Sampling steps
                guidance_scale=7.0,  # CFG guidance strength
            ),
        ),
    )
```

:::

Additional optional configurations:

```python
@neko_cfg
def make_cfg():
    return SD15_lora_train(
        base_model='Lykon/DreamShaper',
        train_steps=1000,
        save_step=200,

        lr=1e-4,
        clip_skip=0,  # Number of skipped clip layers, 0 = none
        low_vram=False,  # Enable 8-bit optimizer

        dataset=dict(
            dataset1=cfg_data_SD_ARB(
                img_root='imgs/',
                batch_size=4,
                resolution=512*512,
                num_bucket=6,
            )
        )
    )
```

::::
:::::

### Training

::::{tab-set}
:::{tab-item} Single-GPU Training

```bash
hcp_train_1gpu --cfg cfgs/train/py/ft1.py
```

You can override config values via CLI:

```bash
hcp_train_1gpu --cfg cfgs/train/py/ft1.py train.train_epochs=10
```
:::

:::{tab-item} Multi-GPU Training

For multi-GPU training, specify GPU IDs and number of GPUs in cfgs/launcher/multi.yaml, then run:

```bash
hcp_train --cfg cfgs/train/py/ft1.py
```

Override config values via CLI:

```bash
hcp_train --cfg cfgs/train/py/ft1.py train.train_epochs=10
```
:::
::::

After training, all output will be saved in a folder like:

```text
exps/2023-07-26-01-05-35
├── cfg.yaml           # Configuration file
├── ckpts              # LoRA model checkpoints
│   ├── unet-100.safetensors
│   ├── unet-200.safetensors
│   ├── ...
├── tblog              # Tensorboard logs
│   └── events.out.tfevents.1690346085.myenvironment.210494.0
└── train.log
```

### Advanced Configuration

Train specific layers or use layer-wise training:

```python
model_part = CfgWDModelParser([
    dict(
        lr=1e-6,
        # k, v, and ff layers (using regex to specify layer names)
        layers=[
            're:.*\.to_k',
            're:.*\.to_v',
            're:.*\.ff'
        ],
    ),
    dict(
        lr=1e-5,
        layers=[
            're:.*resnets'
        ],
    )
], weight_decay=1e-2),
```

## Fine-Tuning with DreamBooth

To fine-tune with DreamBooth, prepare a regularization dataset and specify it in the config file:

```python
@neko_cfg
def cfg_data():
    return dict(
        # Training dataset
        dataset1=TextImagePairDataset(_partial_=True, batch_size=4, loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root='imgs/',
                    label_file='${.img_root}',
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(
                bucket=RatioBucket,
                word_names={
                    'pt1': '[V]',      # Trigger word
                    'class': 'dog'     # Subject description
                }
            ),
            bucket=RatioBucket.from_files(
                target_area=512*512,
                num_bucket=6,
            ),
            cache=VaeCache(bs=1)
        ),
        # Regularization dataset
        dataset_class=TextImagePairDataset(_partial_=True, batch_size=1, loss_weight=1.0,
            source=dict(
                data_source1=Text2ImageSource(
                    img_root='imgs_db_class/',
                    label_file='${.img_root}',
                    prompt_template='prompt_template/caption.txt',
                ),
            ),
            handler=StableDiffusionHandler(
                bucket=FixedBucket,
                word_names={'class': 'dog'}
            ),
            bucket=FixedBucket(
                target_size=(512, 512),
            ),
            cache=VaeCache(bs=1)
        )
    )
```

```{tip}
$batch_size × loss_weight can be considered the importance of a dataset. A recommended ratio between training and regularization datasets is 4:1.
```

## Generating Images with the Fine-Tuned Model

After training, you can use the fine-tuned model to generate images. Here, we use a simplified template config. For more advanced usage, refer to:
[Workflow Configuration Guide](../user_guides/workflow.md)

Example simplified config:

```python
from hcpdiff.easy.cfg import SD15_t2i_parts
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SD15_t2i_parts(
        pretrained_model='Lykon/DreamShaper',  # Base model
        parts=['unet-100.safetensors'],  # Trained model checkpoint
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        bs=4,
        width=512,
        height=512,
        guidance_scale=7.0
    )
```

Save and run the config to generate images:

```shell
hcp_run --cfg cfgs/workflow/t2i.py
```

### Advanced Configuration

You can also replace the sampler and VAE in the config:

```python
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from hcpdiff.diffusion.sampler import DiffusersSampler

wrapper = SD15Wrapper.from_pretrained(
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
