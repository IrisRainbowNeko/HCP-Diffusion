# 训练加速与优化

## 使用deepspeed训练

> deepspeed不支持windows，如果需要请使用wsl2

首先运行`accelerate config`配置deepspeed:
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

目前提供`zero2.json`和`zero3.json`两种方式，`zero3`支持offload，更省显存，但会更慢，并且可能需要较大内存。

配置之后运行命令开始训练:
```bash
accelerate launch -m hcpdiff.train_deepspeed --cfg cfgs/train/cfg_file.yaml
```