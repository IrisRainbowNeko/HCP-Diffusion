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
import sys

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

from packaging.version import parse

import diffusers.pipelines.stable_diffusion.convert_from_ckpt as convert_from_ckpt
import torch
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    create_vae_diffusers_config,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)
from diffusers.utils.import_utils import compare_versions
from omegaconf import OmegaConf
from transformers import CLIPTextModel

from hcpdiff.ckpt_manager import CkptManagerSafe, CkptManagerPKL

try:
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt as load_sd_ckpt
except:
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt as load_sd_ckpt

def convert_ldm_clip_checkpoint(checkpoint, local_files_only=False):
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=local_files_only)

    keys = list(checkpoint.keys())

    text_model_dict = {}

    add_prefix = 'cond_stage_model.transformer.embeddings.position_ids' in checkpoint

    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            t_key = key[len("cond_stage_model.transformer."):]
            if add_prefix:
                t_key = 'text_model.'+t_key
            text_model_dict[t_key] = checkpoint[key]

    text_model.load_state_dict(text_model_dict)

    return text_model

def convert_ldm_clip_checkpoint_0_18(checkpoint, local_files_only=False, text_encoder=None):
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=local_files_only)

    keys = list(checkpoint.keys())

    text_model_dict = {}

    add_prefix = 'cond_stage_model.transformer.embeddings.position_ids' in checkpoint

    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            t_key = key[len("cond_stage_model.transformer."):]
            if add_prefix:
                t_key = 'text_model.'+t_key
            text_model_dict[t_key] = checkpoint[key]

    text_model.load_state_dict(text_model_dict)

    return text_model

def custom_convert_ldm_vae_checkpoint(checkpoint, config):
    vae_state_dict = checkpoint

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id:[key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id:[key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old":f"down.{i}.block", "new":f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks+1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old":f"mid.block_{i}", "new":f"mid_block.resnets.{i-1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old":"mid.attn_1", "new":"mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks-1-i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old":f"up.{block_id}.block", "new":f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks+1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old":f"mid.block_{i}", "new":f"mid_block.resnets.{i-1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old":"mid.attn_1", "new":"mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint

def sd_vae_to_diffuser(args):
    original_config = OmegaConf.load(args.original_config_file)
    vae_config = create_vae_diffusers_config(original_config, image_size=512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Convert the VAE model.
    if args.from_safetensors:
        checkpoint = CkptManagerSafe().load_ckpt(args.vae_pt_path)
        converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint, vae_config)
    else:
        checkpoint = torch.load(args.vae_pt_path, map_location=device)
        converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint['state_dict'], vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    vae.save_pretrained(args.dump_path)

def convert_ckpt(args):
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
        for k, v in sd.items():
            if k.startswith(k_from):
                new_sd[k_to+k[len(k_from):]] = v
            else:
                new_sd[k] = v
        return new_sd

    if args.controlnet:
        ckpt_manager = CkptManagerSafe() if args.to_safetensors else CkptManagerPKL()

        sd_control = pipe.controlnet.state_dict()
        sd_control = replace('controlnet_cond_embedding.conv_in', 'cond_head.0', sd_control)
        for i in range(3):
            sd_control = replace(f'controlnet_cond_embedding.blocks.{i*2}', f'cond_head.{2+i*4}', sd_control)
            sd_control = replace(f'controlnet_cond_embedding.blocks.{i*2+1}', f'cond_head.{4+i*4}', sd_control)
        sd_control = replace('controlnet_cond_embedding.conv_out', 'cond_head.14', sd_control)
        sd_control = {f'___.{k}':v for k, v in sd_control.items()}  # Add placeholder for plugin
        os.makedirs(args.dump_path, exist_ok=True)
        ckpt_manager._save_ckpt(sd_control, None, None, save_path=os.path.join(args.dump_path,
                                                                               f'controlnet.{"safetensors" if args.to_safetensors else "ckpt"}'))
    else:
        pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)

if __name__ == "__main__":
    diffusers_version = importlib_metadata.version("diffusers")
    if compare_versions(parse(diffusers_version), '>=', '0.18.0'):
        convert_from_ckpt.convert_ldm_clip_checkpoint = convert_ldm_clip_checkpoint_0_18
    else:
        convert_from_ckpt.convert_ldm_clip_checkpoint = convert_ldm_clip_checkpoint

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the checkpoint to convert."
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

    parser.add_argument("--vae_pt_path", default=None, type=str, help="Path to the VAE.pt to convert.")
    args = parser.parse_args()

    if args.vae_pt_path is None:
        convert_ckpt(args)
    else:
        sd_vae_to_diffuser(args)
    # python -m hcpdiff.tools.sd2diffusers --checkpoint_path test/control_sd15_canny.pth --original_config_file test/config.yaml --dump_path test/ckpt/control --controlnet
