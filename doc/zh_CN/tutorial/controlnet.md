# ControlNet 使用教程

## ControlNet 插件配置
ControlNet在这里作为插件来定义，所以在配置文件中，需要在`plugin_unet`中配置ControlNet。
```yaml
plugin_unet:
  controlnet1:
    _target_: hcpdiff.models.controlnet.ControlNetPlugin
    _partial_: True
    lr: 1e-4
    from_layers:
      - 'pre_hook:'
      - 'pre_hook:conv_in' # to make forward inside autocast
    to_layers:
      - 'down_blocks.0'
      - 'down_blocks.1'
      - 'down_blocks.2'
      - 'down_blocks.3'
      - 'mid_block'
      - 'pre_hook:up_blocks.3.resnets.2'
```

## 转换成HCP支持的格式

转换官方模型为`hcp plugin`格式，例如转换`control_v11p_sd15_openpose`:
```yaml
python -m hcpdiff.tools.sd2diffusers --checkpoint_path "ckpts/control_v11p_sd15_openpose.pth" --original_config_file "ckpts/control_v11p_sd15_openpose.yaml" --dump_path "ckpts/control_v11p_sd15_openpose" --controlnet
```

## 训练 ControlNet

示例在`cfgs/train/examples/controlnet.yaml`

### 配置模型训练参数

```yaml
_base_:
  - cfgs/train/train_base.yaml
  - cfgs/plugins/plugin_controlnet.yaml # include controlnet plugin


plugin_unet:
  controlnet1:
    lr: 1e-5 # 修改学习率
```

### 配置数据集

ControlNet训练需要`条件图，文本，内容图`三个一对，所以这里使用`TextImageCondPairDataset`配合`Text2ImageCondSource`。通过`cond_root`指定条件图的文件夹。
```yaml
data:
  dataset1:
    _target_: hcpdiff.data.TextImageCondPairDataset
    _partial_: True # Not directly instantiate the object here. There are other parameters to be added in the runtime.
    batch_size: 4
    cache_latents: True
    att_mask_encode: False
    loss_weight: 1.0

    source:
      data_source1:
        _target_: hcpdiff.data.source.Text2ImageCondSource
        img_root: 'imgs/'
        cond_root: 'cond_imgs/'
        prompt_template: 'prompt_tuning_template/object.txt'
        caption_file: null # path to image captions (file_words)
        att_mask: null
        bg_color: [ 255, 255, 255 ] # RGB; for ARGB -> RGB

        text_transforms:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: hcpdiff.utils.caption_tools.TagShuffle
            - _target_: hcpdiff.utils.caption_tools.TagDropout
              p: 0.1
            - _target_: hcpdiff.utils.caption_tools.TemplateFill
              word_names: { }
    bucket:
      _target_: hcpdiff.data.bucket.RatioBucket.from_files # aspect ratio bucket
      target_area: ${hcp.eval:"512*512"}
      num_bucket: 5
```
`条件图，内容图`分别放到不同文件夹中，文件名相同的是一对。如果使用txt格式的标注文件，最好和`内容图`放到同一个文件夹中。

## 使用 ControlNet 生成图像

示例在`cfgs/infer/img2img_controlnet.yaml`中

### 添加条件图输入

填写条件图的路径
```yaml
ex_input:
  cond:
    _target_: hcpdiff.data.data_processor.ControlNetProcessor
    image: 'cond_img.png'
```

### 配置 ControlNet 插件

```yaml
merge:
  plugin_cfg: cfgs/plugins/plugin_controlnet.yaml # 插件定义文件

  group1:
    type: 'unet'
    base_model_alpha: 1.0
    plugin:
      controlnet1:
        path: 'ckpts/controlnet.ckpt' # ControlNet 模型路径(转换为hcp格式的)
        layers: 'all'
```

