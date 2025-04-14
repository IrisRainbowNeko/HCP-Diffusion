# 训练加速与优化

## 使用deepspeed训练

```{important}
deepspeed不支持windows，如果需要请使用wsl2
```

首先在`cfgs/launcher/deepspeed.yaml`中指定`deepspeed_config.deepspeed_config_file`为需要的deepspeed的配置文件。目前提供`zero2.json`和`zero3.json`两种方式，`zero3`支持offload，更省显存，但会更慢，并且可能需要较大内存。

训练使用的显卡数量通过`num_processes`配置。

配置之后运行命令开始训练:
```bash
hcp_train --launch_cfg cfgs/launcher/deepspeed.yaml --cfg cfgs/train/cfg_file.yaml
```