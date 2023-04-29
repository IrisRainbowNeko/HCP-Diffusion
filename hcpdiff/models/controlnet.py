from typing import List, Tuple, Union, Optional, Dict, Any

from diffusers import UNet2DConditionModel
import torch
from torch import nn
from copy import deepcopy

from .plugin import MultiPluginBlock, BasePluginBlock
from hcpdiff.utils.net_utils import remove_all_hooks, remove_layers

class ControlNetPlugin(MultiPluginBlock):
    def __init__(self, name:str, from_layers: List[Dict[str, Any]], to_layers: List[Dict[str, Any]], host_model: UNet2DConditionModel=None,
                 cond_block_channels=(3, 16, 32, 96, 256, 320),
                 layers_per_block=2, block_out_channels: Tuple[int] = (320, 640, 1280, 1280)):
        super().__init__(name, from_layers, to_layers, host_model=host_model)

        self.register_input_feeder_to(host_model)

        self.conv_in = self.copy_block(host_model.conv_in)
        self.time_proj = self.copy_block(host_model.time_proj)
        self.time_embedding = self.copy_block(host_model.time_embedding)
        self.class_embedding = self.copy_block(host_model.class_embedding)
        self.down_blocks = self.copy_block(host_model.down_blocks)
        self.mid_block = self.copy_block(host_model.mid_block)
        self.dtype = host_model.dtype

        self.build_head(cond_block_channels)

        self.controlnet_down_blocks = nn.ModuleList(
            [nn.Conv2d(block_out_channels[0], block_out_channels[0], kernel_size=1)]+
            [nn.Conv2d(out_ch, out_ch, kernel_size=1) for out_ch in block_out_channels for _ in range(layers_per_block+1)]
        )
        self.controlnet_mid_block = self.controlnet_down_blocks[-1]
        del self.controlnet_down_blocks[-1]

        self.reset_parameters()

    def copy_block(self, block):
        if block is None:
            return block
        block = deepcopy(block)
        remove_all_hooks(block)
        remove_layers(block, BasePluginBlock)
        return block

    def build_head(self, cond_block_channels):
        cond_head = [
            nn.Conv2d(cond_block_channels[0], cond_block_channels[1], kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for i in range(2, (len(cond_block_channels) - 2) * 2):
            cond_head.append(nn.Conv2d(cond_block_channels[i // 2], cond_block_channels[(i + 1) // 2], kernel_size=3, padding=1, stride=1 + i % 2))
            cond_head.append(nn.SiLU())
        cond_head.append(nn.Conv2d(cond_block_channels[-2], cond_block_channels[-1], kernel_size=3, padding=1))
        self.cond_head = nn.Sequential(*cond_head)

    def reset_parameters(self) -> None:
        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
        self.controlnet_down_blocks.apply(weight_init)
        self.controlnet_mid_block.apply(weight_init)
        self.cond_head[-1].apply(weight_init)

    def from_layer_hook(self, host, fea_in:Tuple[torch.Tensor], fea_out:Tuple[torch.Tensor], idx: int):
        if idx==0:
            self.data_input = fea_in
        elif idx==1:
            self.feat_to = self(*self.data_input)

    def to_layer_hook(self, host, fea_in:Tuple[torch.Tensor], fea_out:Tuple[torch.Tensor], idx: int):
        if idx == 5:
            sp = fea_in[0].shape[1]//2
            new_feat = fea_in[0].clone()
            new_feat[:, sp:, ...] = fea_in[0][:, sp:, ...] + self.feat_to[0]
            return (new_feat, fea_in[1])
        elif idx == 3:
            return (fea_out[0], tuple(fea_out[1][i] + self.feat_to[(idx) * 3 + i+1] for i in range(2)))
        elif idx == 4:
            return fea_out + self.feat_to[11+1]
        else:
            return (fea_out[0], tuple(fea_out[1][i]+self.feat_to[(idx)*3+i+1] for i in range(3)))

    def feed_input_data(self, data): # get the control image
        if isinstance(data, dict):
            self.cond = data['cond']

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple:

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.cond_head(self.cond)

        sample += controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample,)

        mid_block_res_sample = self.controlnet_mid_block(sample)

        return controlnet_down_block_res_samples + (mid_block_res_sample,)




