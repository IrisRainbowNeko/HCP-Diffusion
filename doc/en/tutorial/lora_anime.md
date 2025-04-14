# LoRA Training Tutorial (Anime Characters)

This section provides a guide for training anime-style characters.

## Workflow and Principle

For this task, the method recommended by HCP-Diffusion author [IrisRainbowNeko](https://github.com/IrisRainbowNeko) involves training an embedding model and a LoRA model together. During inference (i.e., generating images of anime characters), both the embedding and LoRA models are used simultaneously. This approach helps achieve the desired results and offers more stable performance than traditional LoRA methods, as the trigger word is embedded directly into the embedding model.

Thus, the overall training process consists of the following steps:

- Prepare the dataset
- Create the embedding
- Train the model
- Perform inference
- Save in SD WebUI format

## Preparing the Dataset

The first step is to prepare the dataset. You need a set of images with identical dimensions (PNG format is recommended for better training quality), and each image should have a corresponding text label (in .txt format). The final dataset should look like the following structure (in this case, the dataset is stored in /data/surtr_dataset, and each image is 512x704):

```text
/data/surtr_dataset
├── 000834cb567b675eb0904436b2d6dabdb5b09493.png
├── 000834cb567b675eb0904436b2d6dabdb5b09493.txt
├── 0095c8ff0ccaf9ab30c705d9babef91800042497.png
├── 0095c8ff0ccaf9ab30c705d9babef91800042497.txt
├── 00e73bb48d5a2dded1464a433d619f901ee07d6a.png
├── 00e73bb48d5a2dded1464a433d619f901ee07d6a.txt
├── ......
```

```{tip}
A recommended tool for this step is the [waifuc](https://github.com/deepghs/waifuc) project, which can automatically crawl, clean, process, and label images from over a dozen image sites (such as Pixiv, Danbooru, Zerochan, etc.) with just the English name of the character.
```

## Creating the Embedding

To improve the stability of trigger words during image generation, this guide uses an embedding (similar to Textual Inversion), where each embedding represents a keyword.

Use the following command to create an embedding:

```shell
python -m hcpdiff.tools.create_embedding <pretrained_model_path> <word_name> <num_vectors_per_token> [--init_text <initial_word>]
```

For example, if you're training the character Surtr (with the keyword name: surtr_arknights), use:

```shell
python -m hcpdiff.tools.create_embedding deepghs/animefull-latest surtr_arknights 4
```

This will generate a file named surtr_arknights.pt in the embs directory.

## Model Training

Once the preparations are complete, you can begin training.

First, install TensorBoard to monitor training progress in real-time:

```shell
pip install tensorboard
```

::::{tab-set}
:::{tab-item} Single-GPU Training

For single-GPU environments, use the following command:

```shell
hcp_train_1gpu --cfg cfgs/train/py/examples/lora_preview.py \
    model.wrapper.models.ckpt_path=deepghs/animefull-latest \  # Base model
    data_train.dataset1.handler.word_names.pt1=surtr_arknights \  # Trigger word
    data_train.dataset1.source.data_source1.img_root=/data/surtr_dataset  # Dataset path
```

:::

:::{tab-item} Multi-GPU Training

For multi-GPU environments, specify the GPU IDs and number of GPUs in cfgs/launcher/multi.yaml, then run:

```shell
hcp_train --cfg cfgs/train/py/examples/lora_preview.py \
    model.wrapper.models.ckpt_path=deepghs/animefull-latest \  # Base model
    data_train.dataset1.handler.word_names.pt1=surtr_arknights \  # Trigger word
    data_train.dataset1.source.data_source1.img_root=/data/surtr_dataset  # Dataset path
```

:::
::::

```{note}
Details:
- data_train.dataset1.handler.word_names.pt1: The name of the character to be trained, must match the embedding name created earlier. In this case, surtr_arknights.
- data_train.dataset1.source.data_source1.img_root: Path to the dataset, here it's /data/surtr_dataset.
- [Optional] exp_dir: Directory to save experiment data. By default, a subdirectory named with the current timestamp will be created under exps/, e.g., exps/2023-07-26-01-05-35.
- [Optional] train.train_steps: Total training steps. Default is 1000.
- [Optional] train.save_step: Interval for saving model checkpoints. Default is 100 (i.e., save every 100 steps).
- [Optional] model.wrapper.models.ckpt_path: Base diffusion model used for training. Default is deepghs/animefull-latest, which is a leaked official NovelAI model (~7GB). This is a general-purpose anime model and will be automatically downloaded from Hugging Face if not present locally.
```

After training, you will get an experiment directory like this:

```text
exps/2023-07-26-01-05-35
├── cfg.yaml
├── ckpts
│   ├── surtr_arknights-1000.pt
│   ├── surtr_arknights-100.pt
│   ├── ...
│   ├── text_encoder-1000.safetensors
│   ├── ...
│   ├── unet-1000.safetensors
│   ├── ...
├── tblog
│   └── events.out.tfevents.1690346085.myenvironment.210494.0
└── train.log
```

```{note}
Explanation:
- surtr_arknights-xxx.pt: The trained embedding file.
- text_encoder-xxx.safetensors and unet-xxx.safetensors: The trained LoRA model files. (Note: In HCP-Diffusion, LoRA models are split into two parts. To convert them into a format compatible with WebUI, see the Model File Format Guide.)
```

## Model Inference

After training, you can use the trained model to generate images.

::::{tab-set}
:::{tab-item} Using a Configuration File

Copy cfgs/workflow/easy/t2i_lora.py to cfgs/workflow/easy/t2i_lora_surtr.py and modify the configuration:

```python
@neko_cfg
def make_cfg():
    return SD15_t2i_lora(
        pretrained_model='stablediffusionapi/anything-v5',  # Replace base model
        lora_info=[
            ('exps/2023-07-26-01-05-35/lora1-1000.safetensors', 0.8),  # (LoRA model path, weight)
        ],
        prompt='masterpiece, best quality, 1girl, solo, {surtr_arknights-1000:1.2}',  # Prompt
        bs=4,
        width=512,
        height=768,
        guidance_scale=7.5
    )
```

```{note}
Details:
- lora_info: Add the trained LoRA model path and its weight. Adjust the weight as needed.
- prompt: Prompt for image generation. When using an embedding trigger word, use the format character_name-xxxx, where xxxx is the training step number. In this example: surtr_arknights-1000.
- [Optional] negative_prompt: Negative prompt for image generation. Defaults to a general-purpose negative prompt.
- [Optional] N_repeats: Prompt capacity. Default is 1. Increase if prompt is too long and causes errors.
- [Optional] pretrained_model: Base model used for generation. Default is stablediffusionapi/anything-v5, which performs better than deepghs/animefull-latest for anime images.
- [Optional] width: Image width (must be a multiple of 8). Default is 512.
- [Optional] height: Image height (must be a multiple of 8). Default is 768.
- [Optional] guidance_scale: Higher value increases prompt influence, resulting in more consistent images. Default is 7.5.
- [Optional] N_steps: Number of inference steps. Default is 20.
- [Optional] bs: Number of images to generate. Default is 4.
- [Optional] seed: Random seed. Using the same seed with the same settings produces identical images. If not specified, a random seed is used and logged in the Python config file.
- [Optional] save_root: Output directory for generated images. Default is output_pipe/.
```

Run the following command to generate images:

```shell
hcp_run --cfg cfgs/workflow/easy/t2i_lora_surtr.yaml
```

:::

:::{tab-item} Using CLI Arguments with Predefined Config

```shell
hcp_run --cfg cfgs/workflow/easy/t2i_lora_cli.yaml \
    lora_path=exps/2023-07-26-01-05-35/lora1-1000.safetensors \
    prompt='masterpiece, best quality, 1girl, solo, {surtr_arknights-1000:1.2}'
```

:::
::::

After execution, a PNG image and a Python config file will be generated in the output directory. These represent the generated image and the configuration used, respectively.

An example image might look like this (note that your result may vary due to the random seed):

![surtr_arknight_sample](../../imgs/surtr_arknights_sample.png)

## Using the Model in WebUI

To use the trained LoRA model in a1111's WebUI, you need to save it in a compatible format. For details, refer to the [LoRA Training Guide](./lora.md).

The WebUI-compatible LoRA model files will be located in:

- exps/2023-07-26-01-05-35/ckpts/

If you want to publish the model on civitai.com, upload the following files:

- exps/2023-07-26-01-05-35/ckpts/lora_webui-1000.safetensors – the LoRA model file
- exps/2023-07-26-01-05-35/ckpts/surtr_arknights-1000.pt – the embedding trigger word file

Once both models are loaded in WebUI, you’ll be able to generate beautiful anime waifus with ease! ✨