# Image Generation

HCP-Diffusion generates images based on a `workflow` system. The workflow is defined in a configuration file, which is a standard `Python` (.py) file. This allows you to describe the image generation process programmatically. Using workflows, you can incorporate various operations into the generation process, such as super-resolution, localized editing, and more. You can even assign different prompts, CFG scales, or models to each step in the workflow.

```bash
# Run the workflow
hcp_run --cfg cfgs/workflow/text2img.yaml
```

## Adjust Word Attention

```{note}
You can emphasize specific words or phrases in the prompt during image generation:

Format: {text_to_emphasize:multiplier}, with a default multiplier of 1.1.

Example:
a {cat} running {in the {city}:1.2}

In this case:
- "cat" is emphasized by 1.1x
- "in the" is emphasized by 1.2x
- "city" is emphasized by 1.2 * 1.1 = 1.32x
```

## Basic Configuration Structure

The entry point for the workflow is the make_cfg function. The returned dictionary must contain a workflow key that defines the workflow.

### Image Generation with Stable Diffusion

:::::{tab-set}

::::{tab-item} Simplified Configuration (Beginner-Friendly)

The file cfgs/workflow/easy/text2img.yaml provides a simplified configuration for image generation. With just a few parameter settings, you can generate images easily. However, it offers limited flexibility and fewer features.

Common configuration:

```python
from hcpdiff.easy.cfg import SD15_t2i
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SD15_t2i(
        pretrained_model='Lykon/DreamShaper',  # Path to the pretrained model
        prompt='masterpiece, best quality, 1girl, cat ears, outside',  # Positive prompt
        # negative_prompt='',  # Optional: Negative prompt
        bs=4,  # Batch size
        width=512,  # Image width
        height=512,  # Image height
        guidance_scale=7.0  # CFG guidance scale
    )
```

To change the sampler, you can set the noise_sampler parameter (default is dpmpp_2m_karras):

```python
from hcpdiff.easy import Diffusers_SD

SD15_t2i(
        pretrained_model='Lykon/DreamShaper',
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        bs=4,
        width=512,
        height=512,
        guidance_scale=7.0,
        noise_sampler=Diffusers_SD.euler_a  # Replace sampler
    )
```

Available samplers:

| Sampler         | Description                            |
|-----------------|----------------------------------------|
| dpmpp_2m        | Fewer steps                            |
| dpmpp_2m_karras | Fewer steps, high quality, commonly used |
| ddim            | More steps                             |
| euler           |                                        |
| euler_a         | Common in anime-style, smoother output |

Other configurable options:

```python
from hcpdiff.easy.cfg import SD15_t2i
from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return SD15_t2i(
        pretrained_model='Lykon/DreamShaper',
        prompt='masterpiece, best quality, 1girl, cat ears, outside',
        # negative_prompt='',
        bs=4,
        width=512,
        height=512,
        guidance_scale=7.0,

        seed=42,  # Set random seed
        N_steps=30,  # Number of sampling steps
        save_root='output_pipe/',  # Output directory
    )
```

::::

::::{tab-item} Full Configuration (For Advanced Users)

The file cfgs/workflow/text2img.yaml provides a more flexible and feature-rich configuration for image generation. The text-to-image workflow typically consists of several modules. For more examples, refer to other files in the cfgs/workflow/ directory.

:::{dropdown} Model Loading
:animate: fade-in

```python
import torch
from rainbowneko.parser import neko_cfg
from hcpdiff.easy import Diffusers_SD, SD15_auto_loader
from rainbowneko.infer import Actions, PrepareAction
from hcpdiff.workflow import BuildModelsAction

@neko_cfg
def build_model(pretrained_model='ckpts/any5', noise_sampler=Diffusers_SD.dpmpp_2m_karras) -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=torch.float16),  # Set device and precision
        BuildModelsAction(  # Build and load pretrained model
            model_loader=SD15_auto_loader(_partial_=True,
                ckpt_path=pretrained_model,
                noise_sampler=noise_sampler  # Set sampler (preset)
            )
        ),
    ])
```

````{note}
The noise_sampler here uses a preset configuration. Available presets:

| Sampler         | Description                            |
|-----------------|----------------------------------------|
| dpmpp_2m        | Fewer steps                            |
| dpmpp_2m_karras | Fewer steps, high quality, commonly used |
| ddim            | More steps                             |
| euler           |                                        |
| euler_a         | Common in anime-style, smoother output |

For custom sampler configurations, use full setup like below:

```python
from diffusers import DPMSolverMultistepScheduler
from hcpdiff.diffusion.sampler import DiffusersSampler

# Use a Diffusers sampler
noise_sampler=DiffusersSampler(
    DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        algorithm_type='sde-dpmsolver++',
        use_karras_sigmas=True,
    )
)
```
````

:::

:::{dropdown} Model Optimization
:animate: fade-in

```python
from hcpdiff.workflow import PrepareDiffusionAction, XformersEnableAction, VaeOptimizeAction

@neko_cfg
def optimize_model() -> Actions:
    return Actions([
        PrepareDiffusionAction(),  # Set model-specific parameters
        XformersEnableAction(),  # Enable xformers to optimize memory
        VaeOptimizeAction(slicing=True),  # VAE optimization to reduce memory usage
    ])
```

:::

:::{dropdown} Text Encoding
:animate: fade-in

```python
from hcpdiff.workflow import TextHookAction, AttnMultTextEncodeAction

@neko_cfg
def text(prompt=prompt, negative_prompt=negative_prompt, bs=4, N_repeats=1, layer_skip=1) -> Actions:
    return Actions([
        # Advanced text encoder support
        # N_repeats: Extend max prompt length
        # layer_skip: Skip last N layers (clip skip), note: 0 here equals 1 in webui
        TextHookAction(N_repeats=N_repeats, layer_skip=layer_skip),
        # Text encoding with token weighting support
        AttnMultTextEncodeAction(
            prompt=prompt,  # Positive prompt
            negative_prompt=negative_prompt,  # Negative prompt
            bs=bs  # Batch size
        ),
    ])
```

:::

:::{dropdown} Diffusion Configuration
:animate: fade-in

```python
from hcpdiff.workflow import SeedAction, MakeTimestepsAction, MakeLatentAction

@neko_cfg
def config_diffusion(seed=42, N_steps=20, width=512, height=512) -> Actions:
    return Actions([
        SeedAction(seed),  # Set random seed
        MakeTimestepsAction(N_steps=N_steps),  # Set number of sampling steps
        # MakeTimestepsAction(N_steps=N_steps, strength=0.6),  # Denoising strength (for img2img)
        MakeLatentAction(width=width, height=height)  # Set image dimensions
    ])
```

:::

:::{dropdown} Image Generation
:animate: fade-in

```python
from rainbowneko.infer import LoopAction
from hcpdiff.workflow import DiffusionStepAction, time_iter

@neko_cfg
def diffusion(guidance_scale=7.0) -> Actions:
    return Actions([
        LoopAction(  # Loop through actions
            iterator=time_iter,  # Time step iterator [{'t':t} for t in timesteps]
            actions=[
                DiffusionStepAction(guidance_scale=guidance_scale)  # Perform one denoising step
            ]
        )
    ])
```

:::

:::{dropdown} Image Decoding
:animate: fade-in

```python
from hcpdiff.workflow import DecodeAction, SaveImageAction

@neko_cfg
def decode() -> Actions:
    return Actions([
        DecodeAction(),  # Decode latent to image using VAE
        SaveImageAction(save_root='output_pipe/', image_type='png'),  # Save image
    ])
```

:::

::::
:::::

## Supported Actions

TODO