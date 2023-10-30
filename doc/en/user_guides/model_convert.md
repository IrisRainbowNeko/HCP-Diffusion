# Convert webui format model

This tutorial is about how to convert public models in webUI format to and from the format within this framework,
including base models and lora models.

## Base model conversion

The basic models downloaded directly are usually in the form of a single file format such as ```.ckpt``` or ```.safetensors```.
We need to convert them into the diffuser file format.

+ Download the [config file](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-inference.yaml)
+ Convert models based on config file

```bash
python -m hcpdiff.tools.sd2diffusers \
    --checkpoint_path "path_to_stable_diffusion_model" \
    --original_config_file "path_to_config_file" \
    --dump_path "save_directory" \
    [--extract_ema] # Extract ema model
    [--from_safetensors] # Whether the original model is in safetensors format
    [--to_safetensors] # Whether to save to safetensors format
```

## lora model conversion

### convert form webUI
The lora models used in webUI cannot be directly loaded and need to be converted into the format of this framework.
After conversion, they are divided into two weight files: ```unet``` and ```text_encoder```.

```bash
python -m hcpdiff.tools.lora_convert --from_webui --lora_path lora.safetensors --dump_path lora_hcp/ \
      --auto_scale_alpha # auto scale alpha to be compatible with webui models
```

### convert to webUI
You can also convert the lora models trained within this framework into the webUI format:

```bash
python -m hcpdiff.tools.lora_convert --to_webui --lora_path unet-lora.safetensors --lora_path_TE text_encoder-lora.safetensors --dump_path lora-webui.safetensors \
      --auto_scale_alpha # auto scale alpha to be compatible with webui models
```

## vae model conversion

This framework also supports loading VAE models separately.
The directly downloaded VAE models are in the form of a single file format such as ```.pt``` or ```.safetensors```, which need to be converted as well:

```bash
python -m hcpdiff.tools.sd2diffusers \
    --vae_pt_path "path_to_VAE_model" \
    --original_config_file "path_to_config_file" \
    --dump_path "save_directory"
    [--from_safetensors]
```

## example

1. Download a base model [counterfeit-v30](https://civitai.com/models/4468/counterfeit-v30) from [civitai](https://civitai.com/) and its preview as follows:

<img src="../imgs/CounterfeitV30_sample.jpeg" style="zoom: 30%">

2. Convert the base model.
3. Download a lora model [akira-ray-v10](https://civitai.com/models/34147/akira-ray-nijisanji)
4. Convert the lora model.
5. Perform inference. Note that the paths to the base model and the converted lora model need to be replaced.

```bash
python -m hcpdiff.visualizer --cfg cfgs/infer/webui_model_infer.yaml
```

When using the lora model converted from webUI, you need to add the following parameter to the lora config ```alpha_auto_scale: False```:
```yaml
merge: 
  group1:
    type: 'unet'
    base_model_alpha: 1.0
    lora:
      - path: 'unet-100.safetensors'
        alpha: 0.6
        layers: 'all'
        alpha_auto_scale: False # Disable alpha auto-scaling
    part: null
```

Final result:

<img src="../imgs/akira_ray_v10_output.png" style="zoom: 50%">