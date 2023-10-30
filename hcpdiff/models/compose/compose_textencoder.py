"""
compose_textencoder.py
====================
    :Name:        compose textencoder
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     07/08/2023
    :Licence:     Apache-2.0

support for SDXL.
"""
from typing import Dict, Optional, Union, Tuple, List

import torch
from torch import nn
from transformers import CLIPTextModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

class ComposeTextEncoder(PreTrainedModel):
    def __init__(self, model_list: List[Tuple[str, CLIPTextModel]], cat_dim=-1, with_hook=True):
        super().__init__(PretrainedConfig(**{name:model.config for name, model in model_list}))
        self.cat_dim = cat_dim
        self.with_hook = with_hook

        self.model_names = []
        for name, model in model_list:
            setattr(self, name, model)
            self.model_names.append(name)

    def get_input_embeddings(self) -> List[nn.Module]:
        nn.ParameterDict
        return [getattr(self, name).get_input_embeddings() for name in self.model_names]

    def set_input_embeddings(self, value_dict: Dict[str, int]):
        for name, value in value_dict.items():
            getattr(self, name).set_input_embeddings(value)

    def gradient_checkpointing_enable(self):
        for name in self.model_names:
            getattr(self, name).gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, CLIPTextModel

        >>> clip_B = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> clip_bigG = CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        >>> tokenizer_B = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer_bigG = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        >>> clip_model = MultiTextEncoder([('clip_B', clip_B), ('clip_bigG', clip_bigG)])

        >>> inputs = {
        >>>     'clip_B':tokenizer_B(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt").input_ids
        >>>     'clip_bigG':tokenizer_bigG(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt").input_ids
        >>> }

        >>> outputs = clip_model(input_ids=inputs)
        >>> last_hidden_state = outputs.last_hidden_state  # [B,L,768]+[B,L,1280] -> [B,L,2048]
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        input_ids_list = input_ids.chunk(len(self.model_names),dim=-1)

        if self.with_hook:
            encoder_hidden_states_list, pooled_output_list = [], []
            for name, input_ids in zip(self.model_names, input_ids_list):
                encoder_hidden_states, pooled_output = getattr(self, name)(
                    input_ids,  # get token for model self.{name}
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
                encoder_hidden_states_list.append(encoder_hidden_states)
                pooled_output_list.append(pooled_output)
            encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=self.cat_dim)
            return encoder_hidden_states, pooled_output_list
        else:
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            text_feat_list = {'last_hidden_state':[], 'pooler_output':[], 'hidden_states':[], 'attentions':[]}
            for name, input_ids in zip(self.model_names, input_ids_list):
                text_feat: BaseModelOutputWithPooling = getattr(self, name)(
                    input_ids,  # get token for model self.{name}
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
                text_feat_list['last_hidden_state'].append(text_feat.last_hidden_state)
                text_feat_list['pooler_output'].append(text_feat.pooler_output)
                text_feat_list['hidden_states'].append(text_feat.hidden_states)
                text_feat_list['attentions'].append(text_feat.attentions)

            last_hidden_state = torch.cat(text_feat_list['last_hidden_state'], dim=self.cat_dim)
            # pooler_output = torch.cat(text_feat_list['pooler_output'], dim=self.cat_dim)
            pooler_output = text_feat_list['pooler_output']
            if text_feat_list['hidden_states'][0] is None:
                hidden_states = None
            else:
                hidden_states = [torch.cat(states, dim=self.cat_dim) for states in zip(*text_feat_list['hidden_states'])]

            if return_dict:
                return BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooler_output,
                    hidden_states=hidden_states,
                    attentions=text_feat_list['attentions'],
                )
            else:
                return (last_hidden_state, pooler_output)+hidden_states

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: List[Tuple[str, str]], *args,
                        subfolder: Dict[str, str] = None, revision: str = None, **kwargs):
        r"""
            Examples: sdxl text encoder

            ```python
            >>> sdxl_clip_model = ComposeTextEncoder.from_pretrained([
            >>>         ('clip_B',"openai/clip-vit-base-patch32"),
            >>>         ('clip_bigG',"laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
            >>>     ], subfolder={'clip_B':'text_encoder', 'clip_bigG':'text_encoder_2'})
            ```
        """
        clip_list = [(name, CLIPTextModel.from_pretrained(path, subfolder=subfolder[name], **kwargs)) for name, path in pretrained_model_name_or_path]
        compose_model = cls(clip_list)
        return compose_model
