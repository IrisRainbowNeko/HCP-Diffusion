# Model Inference Tutorial

The inference phase also uses the ``.yaml`` configuration file to describe which components to use.

## Inference Configuration

```yaml
pretrained_model: '' # Pre-trained base model
prompt: '' # Text for generation 
#Negative text that excludes undesired features
neg_prompt: 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
out_dir: 'output/' # Images output directory
emb_dir: 'embs/' # Custom word (embedding) directory
N_repeat: 1 # Sentence length extension multiplier
clip_skip: 1
bs: 4 # batch_size
num: 1 # Total number of images = bs*num
seed: null # random seed
fp16: True # Half-precision inference is faster with less VRAM

condition: null # img2img和contorlnet

save:
  save_cfg: True # save configuration
  image_type: png # image format for save
  quality: 95 # Image compression quality for save

infer_args:
  width: 512
  height: 512
  guidance_scale: 7.5 # CFG scale

new_components: {} # Components for replacing models Sampler, VAE, etc.

merge: null # Loading models and plugins such as lora
```

Replace Sampler:
```yaml
new_components:
  scheduler:
    _target_: diffusers.EulerAncestralDiscreteScheduler # change Sampler
    beta_start: 0.00085
    beta_end: 0.012
    beta_schedule: 'scaled_linear'
```

Replace VAE:
```yaml
new_components:
  vae:
    _target_: diffusers.AutoencoderKL.from_pretrained
    pretrained_model_name_or_path: 'any3.0/vae' # path to vae model
```

## Load the trained model
Multiple models can be specified for layer-wise ensemble, or multiple lora can be added.

```yaml
merge:
    exp_dir: '2023-04-03-10-10-36'
    
    group1: # Multiple groups can be loaded at the same time, with different parameters.
      type: 'unet' # Load to unet or text-encoder
      base_model_alpha: 1.0 # base model weight to merge with lora or part
      lora: # A group can load multiple lora models
        - path: 'exps/${....exp_dir}/ckpts/unet-600.ckpt'
          alpha: 0.7 # base_model*base_model_alpha + lora*alpha
          layers: 'all' # Specify which layers to load, "all" is all layers
          mask: [0.5, 1] # batch_mask of lora, [0.5, 1] corresponds to the positive branch of DreamArtist++
        - path: 'exps/${....exp_dir}/ckpts/unet-neg-600.ckpt'
          alpha: 0.55
          layers: 'all'
          mask: [0, 0.5] # [0, 0.5] corresponds to the negative branch of DreamArtist++
      part: null
    
    group2:
      type: 'TE' # Load to text-encoder
      base_model_alpha: 1.0 # base model weight to merge with lora or part
      lora:
        - path: 'exps/${....exp_dir}/ckpts/text_encoder-600.ckpt'
          alpha: 0.7
          layers: 'all'
          mask: [0.5, 1]
        - path: 'exps/${....exp_dir}/ckpts/text_encoder-neg-600.ckpt'
          alpha: 0.55
          layers: 'all'
          mask: [0, 0.5]
      part: null
    
    group3:
      type: 'unet'
      base_model_alpha: 0.0 # Replace layers of the base model
      part: 
        - path: 'exps/${....exp_dir}/ckpts/unet_ft.ckpt'
          alpha: 1.0 # Replace layers of the base model
          layers: 'all'
```

If you use the DreamArtist, you need to include the corresponding trigger words ```{name}``` and ```{name}-neg``` in the prompt and negative_prompt respectively.
If you use the DreamArtist++, adding only positive trigger words in the prompt will give better results.

### Load Plugins

Besides lora, custom plugins also supported. lora can also be added as a custom plugin, like `controlnet`:

First convert the official model to `hcp plugin` format, for example convert `control_v11p_sd15_openpose`:
```yaml
python -m hcpdiff.tools.sd2diffusers --checkpoint_path "ckpts/control_v11p_sd15_openpose.pth" --original_config_file "ckpts/control_v11p_sd15_openpose.yaml" --dump_path "ckpts/control_v11p_sd15_openpose" --controlnet
```

After that add the relevant configurations in the generation config file and load the model weights:
```yaml
merge:
  # Model plugin definition file
  plugin_cfg: cfgs/plugins/plugin_controlnet.yaml
    
  group3:
    type: 'unet'
    base_model_alpha: 1.0
    part: null
    lora: null
    plugin:
      controlnet1: # Plugin name, cannot be duplicated
        path: 'ckpts/controlnet.ckpt' # Plugin weights file
        layers: 'all'
```

## Word attention multiply
It is possible to change the attention of some words individually:
```
format: {text:multiplier}, default if 1.1
e.g.: a {cat} running {in the {city}:1.2}
```
Where ```cat``` is enhanced by 1.1 times, ```in the``` is enhanced by 1.2 times, and ```city``` is enhanced by 1.2*1.1 times.