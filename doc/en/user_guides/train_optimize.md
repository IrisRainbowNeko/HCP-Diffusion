# Training Acceleration and Optimization

## Training with DeepSpeed

```{important}
DeepSpeed is not supported on Windows. If you're using Windows, please use WSL2.
```

First, specify the desired DeepSpeed configuration file by setting deepspeed_config.deepspeed_config_file in cfgs/launcher/deepspeed.yaml. Currently, two configuration files are provided: zero2.json and zero3.json.

- zero2: Standard ZeRO Stage 2 optimization.
- zero3: Supports parameter offloading, which significantly saves GPU memory but may lead to slower training and higher system memory usage.

The number of GPUs used for training is configured via the num_processes parameter.

After configuration, start training with the following command:

```bash
hcp_train --launch_cfg cfgs/launcher/deepspeed.yaml --cfg cfgs/train/cfg_file.yaml
```