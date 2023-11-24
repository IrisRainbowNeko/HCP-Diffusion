from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from functools import partial
from typing import List, Union, Optional, Tuple

import torch
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch import nn

from .blocks import DownBlock, CrossAttnDownBlock, UpBlock, CrossAttnUpBlock, CrossAttnMidBlock

class SDUNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downblock_builders: List[partial[DownBlock]],
                 midblock_builder: partial[CrossAttnMidBlock],
                 upblock_builders: List[partial[UpBlock]],
                 time_proj_builder: partial[Timesteps],
                 time_embedding_builder: partial[TimestepEmbedding],
                 dropout: float = 0.0,
                 conv_in_kernel: int = 3,
                 time_embedding_dim: int = None, ):
        super().__init__()
        block0_channels = self.downblock_builders[0].keywords["in_channels"]

        conv_in_padding = (conv_in_kernel-1)//2
        self.conv_in = nn.Conv2d(in_channels, block0_channels, kernel_size=conv_in_kernel, padding=conv_in_padding)

        # time embedding
        time_embed_dim = time_embedding_dim or block0_channels*4
        self.time_proj = time_proj_builder(block0_channels)
        timestep_input_dim = block0_channels
        self.time_embedding = time_embedding_builder(timestep_input_dim, time_embed_dim)

        # blocks
        self.downblocks = nn.ModuleList([builder(temb_channels=time_embed_dim, dropout=dropout) for builder in downblock_builders])
        self.midblock = midblock_builder(temb_channels=time_embed_dim, dropout=dropout)
        self.upblocks = nn.ModuleList([builder(temb_channels=time_embed_dim, dropout=dropout) for builder in upblock_builders])

        # proj out
        self.conv_norm_out = nn.GroupNorm(num_channels=block0_channels, num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU
        self.conv_out = nn.Conv2d(block0_channels, out_channels, kernel_size=3, padding=1)

    def encode_timesteps(self, sample: torch.FloatTensor, timesteps: Union[torch.Tensor, float, int]):
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)
        return emb

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple:

        if attention_mask is not None:
            attention_mask = (1-attention_mask.to(sample.dtype))*-10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1-encoder_attention_mask.to(sample.dtype))*-10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. time
        emb = self.encode_timesteps(sample, timestep)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for block in self.downblocks:
            sample, res_samples = block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.midblock(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

        # 5. up
        for i, block in enumerate(self.upblocks):
            is_final_block = i == len(self.up_blocks)-1

            res_samples = down_block_res_samples[-len(block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block:
                upsample_size = down_block_res_samples[-1].shape[2:]

            sample = block(
                hidden_states=sample,
                res_hidden_states_tuple=res_samples,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                upsample_size=upsample_size,
            )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return (sample,)

def unet_sd15(heads=8, cross_attention_dim=768):
    downblock_builders=[
        partial(CrossAttnDownBlock, in_channels=320, out_channels=320, num_layers=2, heads=heads, cross_attention_dim=cross_attention_dim),
        partial(CrossAttnDownBlock, in_channels=320, out_channels=640, num_layers=2, heads=heads, cross_attention_dim=cross_attention_dim),
        partial(CrossAttnDownBlock, in_channels=640, out_channels=1280, num_layers=2, heads=heads, cross_attention_dim=cross_attention_dim),
        partial(DownBlock, in_channels=1280, out_channels=1280, num_layers=2, heads=8, add_downsample=False),
    ]
    midblock_builder = partial(CrossAttnMidBlock, in_channels=1280, out_channels=1280, num_layers=1, heads=heads, cross_attention_dim=cross_attention_dim)
    upblock_builders = [
        partial(UpBlock, in_channels=1280, out_channels=1280, num_layers=3, heads=heads, add_upsample=False),
        partial(CrossAttnUpBlock, in_channels=1280, out_channels=640, num_layers=3, heads=heads, cross_attention_dim=cross_attention_dim),
        partial(CrossAttnUpBlock, in_channels=640, out_channels=320, num_layers=3, heads=heads, cross_attention_dim=cross_attention_dim),
        partial(CrossAttnUpBlock, in_channels=320, out_channels=320, num_layers=3, heads=heads, cross_attention_dim=cross_attention_dim),
    ]

    time_proj_builder = partial(Timesteps, flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedding_builder = partial(TimestepEmbedding, act_fn='silu')

    return SDUNet(4,4,downblock_builders, midblock_builder, upblock_builders, time_proj_builder, time_embedding_builder)