from .compose_textencoder import ComposeTextEncoder
from .compose_tokenizer import ComposeTokenizer
from transformers import CLIPTextModel, AutoTokenizer, CLIPTextModelWithProjection, T5EncoderModel
from typing import Optional, Union, Tuple, Dict
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

class T5EncoderModel_Align(T5EncoderModel):
    # fxxk the transformers!
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutputWithPooling]:
        text_outputs = super().forward(input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
        return BaseModelOutputWithPooling(
            last_hidden_state=text_outputs.last_hidden_state,
            pooler_output=None,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )

class FluxTextEncoder(ComposeTextEncoder):
    def forward(
        self,
        input_ids: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.with_hook:
            encoder_hidden_states_dict, pooled_output_dict = output
            encoder_hidden_states = encoder_hidden_states_dict['T5']
            pooled_output = pooled_output_dict['clip']
        else:
            last_hidden_state = output['last_hidden_state']['T5']
            pooler_output = output['pooler_output']['clip']
            attentions = output['attentions']['T5']
            hidden_states = output['hidden_states']['T5']
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=hidden_states,
                attentions=attentions,
            )

        return encoder_hidden_states, pooled_output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, subfolder=None, revision:str=None, **kwargs):
        clip_L = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder', **kwargs)
        T5 = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder_2', **kwargs)
        return cls({'clip': clip_L, 'T5': T5})

class FluxTokenizer(ComposeTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, subfolder=None, revision:str=None, **kwargs):
        clip_L = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer', **kwargs)
        T5 = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer_2', **kwargs)
        return cls({'clip': clip_L, 'T5': T5})