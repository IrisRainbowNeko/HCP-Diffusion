from hcpdiff.workflow import *
from hcpdiff.workflow.base import feedback_input
from diffusers import EulerAncestralDiscreteScheduler
import torch

dtype = 'fp32'
amp = 'fp32'

memory = {}

@feedback_input
def move_model(memory, **states):
    from hcpdiff.utils.net_utils import to_cpu, to_cuda
    to_cuda(memory.unet)
    to_cuda(memory.text_encoder)
    to_cuda(memory.vae)

def build_model():
    return [
        LoadModelsAction(pretrained_model='ckpts/any5', dtype=dtype,
                         scheduler=EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')),
    ]

def optimize_model():
    return [
        XformersEnableAction(),
        LambdaAction(move_model),
        PrepareDiffusionAction(dtype=dtype, amp=amp),
        VaeOptimizeAction(slicing=True)
    ]

def text(bs=2):
    return [
        TextHookAction(N_repeats=1, layer_skip=1),
        StartTextEncode(),
        FeedInputAction(model=memory.text_encoder),
        AttnMultTextEncodeAction(prompt='masterpiece, best quality, 1girl, cat ears, outside',
                                 negative_prompt='lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
                                 bs=bs),
        EndTextEncode()
    ]

def config_diffusion(seed=42):
    return [
        SeedAction(seed=seed),
        MakeTimestepsAction(N_steps=30),
        MakeLatentAction(width=512, height=512)
    ]

def diffusion():
    return [
        FeedInputAction(model=memory.unet),
        LoopAction(loop_value={'timesteps':'t'},
                   actions=[DiffusionStepAction(guidance_scale=7.0)])
    ]

def decode():
    return [
        DecodeAction(vae=memory.vae),
        SaveImageAction(save_root='output_pipe/', image_type='png')
    ]

@torch.inference_mode()
def run(actions, states):
    N_steps = len(actions)
    for step, act in enumerate(actions):
        print(f'[{step+1}/{N_steps}] action: {type(act).__name__}')
        states = act(memory=memory, **states)
        # print(f'states: {", ".join(states.keys())}')
    return states

if __name__ == '__main__':
    states = {'cfgs': {}}
    run(build_model(), states)
    run(optimize_model(), states)
    run(text(), states)
    run(config_diffusion(), states)
    run(diffusion(), states)
    run(decode(), states)