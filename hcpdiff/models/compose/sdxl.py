from .compose_textencoder import ComposeTextEncoder
from .compose_tokenizer import ComposeTokenizer
from transformers import CLIPTextModel, AutoTokenizer, CLIPTextModelWithProjection
from typing import Optional, Union, Tuple, Dict
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
        try: # old version of transformers
            text_outputs = super().forward(input_ids, attention_mask, position_ids, output_attentions, output_hidden_states, return_dict)
        except TypeError: # new version(like 4.53.1) of transformers removed 'return_dict'
            text_outputs = super().forward(input_ids, attention_mask, position_ids, output_attentions, output_hidden_states)
            
        return BaseModelOutputWithPooling(
            last_hidden_state=text_outputs.last_hidden_state,
            pooler_output=text_outputs.text_embeds,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )

class SDXLTextEncoder(ComposeTextEncoder):
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
            encoder_hidden_states = torch.cat([encoder_hidden_states_dict['clip_L'], encoder_hidden_states_dict['clip_bigG']], dim=-1)
            pooled_output = pooled_output_dict['clip_bigG']
        else:
            last_hidden_state = torch.cat((output['last_hidden_state']['clip_L'], output['last_hidden_state']['clip_bigG']), dim=-1)
            pooler_output = output['pooler_output']['clip_bigG']
            attentions = output['attentions']['clip_bigG']
            if output['hidden_states']['clip_L'] is None:
                hidden_states = None
            else:
                hidden_states = [torch.cat(states, dim=self.cat_dim) for states in zip(output['hidden_states']['clip_L'], output['hidden_states']['clip_bigG'])]
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
        clip_bigG = CLIPTextModelWithProjection_Align.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder_2', **kwargs)
        return cls({'clip_L': clip_L, 'clip_bigG': clip_bigG})

class SDXLTokenizer(ComposeTokenizer):
    def __call__(self, text, *args, max_length=None, **kwargs):
        token_info = super().__call__(text, *args, max_length=max_length, **kwargs)
        token_info['attention_mask'] = token_info['attention_mask']['clip_L']
        return token_info

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, subfolder=None, revision:str=None, **kwargs):
        clip_L = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer', **kwargs)
        clip_bigG = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer_2', **kwargs)
        return cls({'clip_L': clip_L, 'clip_bigG': clip_bigG})
