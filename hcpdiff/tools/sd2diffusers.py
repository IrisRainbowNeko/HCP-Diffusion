# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conversion script for the LDM checkpoints. """

import argparse
import os.path

import torch
from transformers import CLIPTextModel
from hcpdiff.ckpt_manager import CkptManagerSafe, CkptManagerPKL

import diffusers.pipelines.stable_diffusion.convert_from_ckpt as convert_from_ckpt
try:
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt as load_sd_ckpt
except:
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt as load_sd_ckpt

def convert_ldm_clip_checkpoint(checkpoint):
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    keys = list(checkpoint.keys())

    text_model_dict = {}

    add_prefix = 'cond_stage_model.transformer.embeddings.position_ids' in checkpoint

    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            t_key = key[len("cond_stage_model.transformer.") :]
            if add_prefix:
                t_key = 'text_model.'+t_key
            text_model_dict[t_key] = checkpoint[key]

    text_model.load_state_dict(text_model_dict)

    return text_model

if __name__ == "__main__":
    convert_from_ckpt.convert_ldm_clip_checkpoint = convert_ldm_clip_checkpoint

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    # !wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="pndm",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--pipeline_type",
        default=None,
        type=str,
        help=(
            "The pipeline type. One of 'FrozenOpenCLIPEmbedder', 'FrozenCLIPEmbedder', 'PaintByExample'"
            ". If `None` pipeline will be automatically inferred."
        ),
    )
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help=(
            "The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2"
            " Base. Use 768 for Stable Diffusion v2."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        default=None,
        type=str,
        help=(
            "The prediction type that the model was trained on. Use 'epsilon' for Stable Diffusion v1.X and Stable"
            " Diffusion v2 Base. Use 'v_prediction' for Stable Diffusion v2."
        ),
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    parser.add_argument(
        "--upcast_attention",
        action="store_true",
        help=(
            "Whether the attention computation should always be upcasted. This is necessary when running stable"
            " diffusion 2.1."
        ),
    )
    parser.add_argument(
        "--from_safetensors",
        action="store_true",
        help="If `--checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.",
    )
    parser.add_argument(
        "--to_safetensors",
        action="store_true",
        help="Whether to store pipeline in safetensors format or not.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--stable_unclip",
        type=str,
        default=None,
        required=False,
        help="Set if this is a stable unCLIP model. One of 'txt2img' or 'img2img'.",
    )
    parser.add_argument(
        "--stable_unclip_prior",
        type=str,
        default=None,
        required=False,
        help="Set if this is a stable unCLIP txt2img model. Selects which prior to use. If `--stable_unclip` is set to `txt2img`, the karlo prior (https://huggingface.co/kakaobrain/karlo-v1-alpha/tree/main/prior) is selected by default.",
    )
    parser.add_argument(
        "--clip_stats_path",
        type=str,
        help="Path to the clip stats file. Only required if the stable unclip model's config specifies `model.params.noise_aug_config.params.clip_stats_path`.",
        required=False,
    )
    parser.add_argument(
        "--controlnet", action="store_true", default=None, help="Set flag if this is a controlnet checkpoint."
    )
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    args = parser.parse_args()

    pipe = load_sd_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        image_size=args.image_size,
        prediction_type=args.prediction_type,
        model_type=args.pipeline_type,
        extract_ema=args.extract_ema,
        scheduler_type=args.scheduler_type,
        num_in_channels=args.num_in_channels,
        upcast_attention=args.upcast_attention,
        from_safetensors=args.from_safetensors,
        device=args.device,
        stable_unclip=args.stable_unclip,
        stable_unclip_prior=args.stable_unclip_prior,
        clip_stats_path=args.clip_stats_path,
        controlnet=args.controlnet,
    )

    if args.half:
        pipe.to(torch_dtype=torch.float16)

    def replace(k_from, k_to, sd):
        new_sd = {}
        for k,v in sd.items():
            if k.startswith(k_from):
                new_sd[k_to+k[len(k_from):]]=v
            else:
                new_sd[k]=v
        return new_sd

    if args.controlnet:
        ckpt_manager = CkptManagerSafe() if args.to_safetensors else CkptManagerPKL()

        sd_control = pipe.controlnet.state_dict()
        sd_control = replace('controlnet_cond_embedding.conv_in', 'cond_head.0', sd_control)
        for i in range(3):
            sd_control = replace(f'controlnet_cond_embedding.blocks.{i*2}', f'cond_head.{2+i*4}', sd_control)
            sd_control = replace(f'controlnet_cond_embedding.blocks.{i*2+1}', f'cond_head.{4+i*4}', sd_control)
        sd_control = replace('controlnet_cond_embedding.conv_out', 'cond_head.14', sd_control)
        sd_control = {f'___.{k}':v for k,v in sd_control.items()} # Add placeholder for plugin
        os.makedirs(args.dump_path, exist_ok=True)
        ckpt_manager._save_ckpt(sd_control, None, None, save_path=os.path.join(args.dump_path,
                                    f'controlnet.{"safetensors" if args.to_safetensors else "ckpt"}'))
    else:
        pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)

    #python -m hcpdiff.tools.sd2diffusers --checkpoint_path test/control_sd15_canny.pth --original_config_file test/config.yaml --dump_path test/ckpt/control --controlnet