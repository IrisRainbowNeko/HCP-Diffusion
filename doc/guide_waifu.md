# Guide to Training Anime Waifu LoRA Models

This section provides an introduction to training models for anime characters.

## Process and Principles

For this task, the recommended approach by the author of HCP-Diffusion, [7eu7e7](https://github.com/7eu7d7), is to train an embedding model and a Lora model together. During the actual inference (i.e., generating images of anime characters), both the embedding and Lora models are used simultaneously. This achieves the desired effect and results in a more stable performance compared to traditional Lora, as the trigger words are fixed in the embedding model.

The training process consists of the following steps:
* Prepare the dataset
* Create the embedding
* Train the model
* Model inference
* Model format conversion

## Prepare the Dataset

The first step is to prepare the dataset. We need to gather several images with the same dimensions (preferably in png format) and assign corresponding text labels to each image (using txt format). The dataset should have a structure similar to the following (in this case, the dataset is saved in `/data/surtr_dataset`, and all images have dimensions of 512x704):

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

One recommended way to prepare the dataset is to use the [waifuc](https://github.com/deepghs/waifuc) project. By inputting the English name of a character, it automates the process of crawling, cleaning, processing, and labeling character images from multiple image websites (e.g., pixiv, danbooru, zerochan).

## Creating Embedding

To improve the stability of trigger words during image generation, this training method requires an embedding (similar to Texture Inversion), which can be roughly understood as representing a keyword.

First, we use the following command to create the embedding:

```shell
python -m hcpdiff.tools.create_embedding <pretrained_model_path> <word_name> <word_size> [--init_text <initialization_word>]
```

For example, for the character Surtr (with the keyword name: `surtr_arknights`), we can create the embedding as follows:

```shell
python -m hcpdiff.tools.create_embedding deepghs/animefull-latest surtr_arknights 4
```

Now, the `embs` directory will contain a file named `surtr_arknights.pt`.

## Model Training

After completing the preparations above, we can begin the training process.

First, we need to install Tensorboard to monitor the training progress in real-time:

```shell
pip install tensorboard
```

For running on a single GPU (multiple GPU environments are similar, see README), we can execute the following command to start the training:

```shell
accelerate launch -m hcpdiff.train_ac_single \
    --cfg cfgs/train/examples/lora_anime_character.yaml \
    character_name=surtr_arknights \
    dataset_dir=/data/surtr_dataset
```

Where:
* `character_name` is the name of the character to be trained, which should match the name of the embedding created in the previous section, in this case, `surtr_arknights`.
* `dataset_dir` is the path to the dataset, which should be filled with `/data/surtr_dataset`.
* [Optional] `exp_dir` is the path to save the experimental data. By default, it will create a subpath in `exps` directory named with the current date and time, such as `exps/2023-07-26-01-05-35`.
* [Optional] `train.train_steps` is the total number of training steps, with a default value of `1000`.
* [Optional] `train.save_step` is the interval at which the model is saved during training, with a default value of `100` (i.e., saving the model every 100 steps).
* [Optional] `model.pretrained_model_name_or_path` is the diffusion model used for training, with a default value of `deepghs/animefull-latest`, which is the leaked model from NovelAI and approximately 7GB in size. The model is a general model for training anime characters and will be downloaded from the HuggingFace repository automatically for training.

After training, you will obtain an experimental data path as follows:

```text
exps/2023-07-26-01-05-35
├── cfg.yaml
├── ckpts
│   ├── surtr_arknights-1000.pt
│   ├── surtr_arknights-100.pt
│   ├── surtr_arknights-200.pt
│   ├── surtr_arknights-300.pt
│   ├── surtr_arknights-400.pt
│   ├── surtr_arknights-500.pt
│   ├── surtr_arknights-600.pt
│   ├── surtr_arknights-700.pt
│   ├── surtr_arknights-800.pt
│   ├── surtr_arknights-900.pt
│   ├── text_encoder-1000.safetensors
│   ├── text_encoder-100.safetensors
│   ├── text_encoder-200.safetensors
│   ├── text_encoder-300.safetensors
│   ├── text_encoder-400.safetensors
│   ├── text_encoder-500.safetensors
│   ├── text_encoder-600.safetensors
│   ├── text_encoder-700.safetensors
│   ├── text_encoder-800.safetensors
│   ├── text_encoder-900.safetensors
│   ├── unet-1000.safetensors
│   ├── unet-100.safetensors
│   ├── unet-200.safetensors
│   ├── unet-300.safetensors
│   ├── unet-400.safetensors
│   ├── unet-500.safetensors
│   ├── unet-600.safetensors
│   ├── unet-700.safetensors
│   ├── unet-800.safetensors
│   └── unet-900.safetensors
├── tblog
│   └── events.out.tfevents.1690346085.myenvironment.210494.0
└── train.log
```

Where:
* `surtr_arknights-xxx.pt` is the obtained embedding from training.
* `text_encoder-xxx.safetensors` and `unet-xxx.safetensors` are the trained Lora models. (Note: In the HCP-Diffusion framework, the Lora model is divided into two parts. If you need to convert it into a Lora model format supported by webui

, please refer to the last section, [Model Format Conversion](#model-format-conversion)).

## Model Inference

After completing the training, we use the previously trained models to generate images.

```shell
python -m hcpdiff.visualizer \
    --cfg cfgs/infer/anime/text2img_anime_lora.yaml \
    exp_dir=exps/2023-07-26-01-05-35 \
    model_steps=1000 \
    prompt='masterpiece, best quality, 1girl, solo, {surtr_arknights-1000:1.2}'
```

Here are the details:
* `exp_dir`: The path where the training data is located, which should be consistent with the `exp_dir` used during training.
* `model_steps`: The step number of the Lora model to be loaded. For example, if the value here is `1000`, it will load `text_encoder-1000.safetensors` and `unet-1000.safetensors`.
* `prompt`: The prompt word used to generate images. Please note that when using the trigger words from the embedding, the format should be `character_name-xxxx`, where `xxxx` is the step number, and this value should be consistent with `model_steps`. In this example, it would be `surtr_arknights-1000`.
* 【Optional】`neg_prompt`: The negative prompt word used to generate images. The default value is a common negative prompt word.
* 【Optional】`N_repeats`: The capacity of the prompt word. The default value is `2`, and it can be increased if the prompt word is long and causes an error.
* 【Optional】`pretrained_model`: The base model used for generating images. The default value is `stablediffusionapi/anything-v5`, which has better performance in actual anime image generation than `deepghs/animefull-latest`.
* 【Optional】`infer_args.width`: The width of the generated images, which should be a multiple of 8. The default value is `512`.
* 【Optional】`infer_args.height`: The height of the generated images, which should be a multiple of 8. The default value is `768`.
* 【Optional】`infer_args.guidance_scale`: The scale used during image generation, where a higher value gives more control to the prompt words and leads to more similar generated images. The default value is `7.5`.
* 【Optional】`infer_args.num_inference_steps`: The number of steps used during image generation. The default value is `30`.
* 【Optional】`merge.alpha`: The weight of the Lora model during image generation. The default value is `0.85`.
* 【Optional】`num`: The number of generated images. The default value is `1`.
* 【Optional】`bs`: The batch size used during image generation. The total number of generated images will be `num x bs`. The default value is `1`.
* 【Optional】`seed`: The random seed used during image generation. When using the same seed and other configurations are the same, the generated images will be completely deterministic. If `seed` is not specified, a random seed will be used, and the specific value can be found in the corresponding YAML configuration file of the generated image.
* 【Optional】`output_dir`: The export path for the image files. The default value is `output`.

After running the process, you will find a PNG image and a YAML configuration file generated in the `output` directory. The PNG image represents the generated picture, and the YAML file contains detailed configuration information used during the generation process. An example of the generated image is shown below (please note that the seed is randomly selected, so the actual image may differ from the one shown below, it is for reference only):

![surtr_arknight_sample](../imgs/surtr_arknights_sample.png)

## Model Format Conversion

Once you're satisfied with the generated model, you can export the HCP-format Lora model to a format supported by a1111's webui using the following command:

```shell
python -m hcpdiff.tools.lora_convert --to_webui \
    --lora_path unet-xxxx.safetensors \
    --lora_path_TE text_encoder-xxxx.safetensors \
    --dump_path lora-xxxx.safetensors \
    --auto_scale_alpha # The existing webui model doesn't have alpha auto scaling, so it needs to be converted
```

In this example, the actual command used is as follows:

```shell
python -m hcpdiff.tools.lora_convert --to_webui \
    --lora_path exps/2023-07-26-01-05-35/ckpts/unet-1000.safetensors \
    --lora_path_TE exps/2023-07-26-01-05-35/ckpts/text_encoder-1000.safetensors \
    --dump_path exps/2023-07-26-01-05-35/ckpts/lora-1000.safetensors \
    --auto_scale_alpha
```

The webui version of the Lora model file will be exported to `exps/2023-07-26-01-05-35/ckpts/lora-1000.safetensors`.

If you want to publish the file on civitai.com, you just need to upload the following files:
* `exps/2023-07-26-01-05-35/ckpts/lora-1000.safetensors` - Lora model file
* `exps/2023-07-26-01-05-35/ckpts/surtr_arknights-1000.pt` - Embedding trigger word file

On the webui, as long as you use these two models simultaneously, you can draw your anime waifu~~~


