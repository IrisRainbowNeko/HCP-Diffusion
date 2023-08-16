from .compose_textencoder import ComposeTextEncoder
from .compose_tokenizer import ComposeTokenizer
from transformers import CLIPTextModel, AutoTokenizer

class SDXLTextEncoder(ComposeTextEncoder):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, subfolder=None, revision:str=None, **kwargs):
        clip_B = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder')
        clip_bigG = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder_2')
        return cls([('clip_B', clip_B), ('clip_bigG', clip_bigG)])

class SDXLTokenizer(ComposeTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, subfolder=None, revision:str=None, **kwargs):
        clip_B = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer')
        clip_bigG = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer_2')
        return cls([('clip_B', clip_B), ('clip_bigG', clip_bigG)])