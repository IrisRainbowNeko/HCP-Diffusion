# 图像生成 (基于workflow)

使用`workflow`来生成图像会比`visualizer`更加灵活。`workflow`的配置文件同样是一个`.yaml`文件，可以通过`yaml`文件来描述生成图像的过程。使用`workflow`可以在生成过程中加入超分，局部修改等各种操作。甚至可以让每个`step`都使用不同的`prompt`，CFG强度，或使用不同的模型。

```bash
# 运行workflow
python -m hcpdiff.infer_workflow --cfg cfgs/workflow/highres_fix.yaml
```

## 配置基础结构
`workflow`的配置文件分为三个部分:
```yaml
# 储存一些全局对象。比如unet和vae等。
memory: {}

# 准备阶段运行的Action，比如加载unet等模型到memory，或者为模型添加优化和hook。
# 以及加载一些下面的actions需要的，将memory中的对象作为参数的对象。
prepare:
    - ...
    - ...

# 图片生成的workflow，包括参数构建到最终图像存储的整个流程。
actions:
    - ...
    - ...
```



## 支持的Action

### ExecAction
```yaml
# 运行一段python代码，比如把模型放到CUDA
- _target_: hcpdiff.workflow.ExecAction
  prog: |-
    from hcpdiff.utils.net_utils import to_cpu, to_cuda
    to_cuda(memory.unet)
    to_cuda(memory.text_encoder)
    to_cuda(memory.vae)
```

### LoadModelsAction
```yaml
- _target_: hcpdiff.workflow.LoadModelsAction
  pretrained_model: '模型路径'
  dtype: # 支持 [fp32, amp, fp16, bf16]
  scheduler: [可选]
  unet: [可选]
  text_encoder: [可选]
  tokenizer: [可选]
  vae: [可选]
```

### LoopAction
根据loop_value的`{key:value}`迭代 `for value1, value2 in zip(key1, key2)`
```yaml
- _target_: hcpdiff.workflow.LoopAction
  loop_value:
    timesteps: t #迭代timesteps，每一步的存成t到states里
  actions: # 每次迭代运行哪些action
    - _target_: hcpdiff.workflow.DiffusionStepAction
      guidance_scale: 7.0
```