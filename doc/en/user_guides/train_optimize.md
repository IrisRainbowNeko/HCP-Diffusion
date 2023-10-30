# Training Acceleration and Optimization

## Training with DeepSpeed

> DeepSpeed is not supported on Windows. If needed, please use WSL2.

Start by configuring DeepSpeed using `accelerate config`:
```bash
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]:
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: yes
Please enter the path to the json DeepSpeed config file: cfgs/zero2.json
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: 
How many GPU(s) should be used for distributed training? [1]:
```

Currently, two options are provided for DeepSpeed configuration: `zero2.json` and `zero3.json`. `zero3` supports offload, which saves GPU memory but may result in slower training and may require larger memory.

After configuring, start training with the following command:
```bash
accelerate launch -m hcpdiff.train_deepspeed --cfg cfgs/train/cfg_file.yaml
```