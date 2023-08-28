from .compose_textencoder import ComposeTextEncoder
from .compose_tokenizer import ComposeTokenizer
from transformers import CLIPTextModel, AutoTokenizer, CLIPTextModelWithProjection
from typing import Optional, Union, Tuple
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

class CLIPTextModelWithProjection_Align(CLIPTextModelWithProjection):
    # fxxk the transformers!
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        text_outputs = super().forward(input_ids, attention_mask, position_ids, output_attentions, output_hidden_states, return_dict)
        return BaseModelOutputWithPooling(
            last_hidden_state=text_outputs.last_hidden_state,
            pooler_output=text_outputs.text_embeds,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )

class SDXLTextEncoder(ComposeTextEncoder):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, subfolder=None, revision:str=None, **kwargs):
        clip_B = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder', **kwargs)
        clip_bigG = CLIPTextModelWithProjection_Align.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder_2', **kwargs)
        return cls([('clip_B', clip_B), ('clip_bigG', clip_bigG)])

class SDXLTokenizer(ComposeTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, subfolder=None, revision:str=None, **kwargs):
        clip_B = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer', **kwargs)
        clip_bigG = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer_2', **kwargs)
        return cls([('clip_B', clip_B), ('clip_bigG', clip_bigG)])