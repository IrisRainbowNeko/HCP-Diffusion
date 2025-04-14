import random
from typing import Dict, Union, List

import numpy as np
from string import Formatter
from rainbowneko.data import DataHandler
from rainbowneko._share import register_model_callback

class TagShuffleHandler(DataHandler):
    def __init__(self, key_map_in=('prompt -> prompt',), key_map_out=('prompt -> prompt',)):
        super().__init__(key_map_in, key_map_out)

    def handle(self, prompt: Union[Dict[str, str], str]):
        if isinstance(prompt, str):
            tags = prompt.split(',')
            random.shuffle(tags)
            prompt = ','.join(tags)
        else:
            tags = prompt['caption'].split(',')
            random.shuffle(tags)
            prompt['caption'] = ','.join(tags)
        return {'prompt':prompt}

    def __repr__(self):
        return 'TagShuffleHandler()'

class TagDropoutHandler(DataHandler):
    def __init__(self, p=0.1, key_map_in=('prompt -> prompt',), key_map_out=('prompt -> prompt',)):
        super().__init__(key_map_in, key_map_out)
        self.p = p

    def handle(self, prompt: Union[Dict[str, str], str]):
        if isinstance(prompt, str):
            tags = np.array(prompt.split(','))
            prompt = ','.join(tags[np.random.random(len(tags))>self.p])
        else:
            tags = prompt['caption'].split(',')
            prompt['caption'] = ','.join(tags[np.random.random(len(tags))>self.p])
        return {'prompt':prompt}

    def __repr__(self):
        return f'TagDropoutHandler(p={self.p})'

class TagEraseHandler(DataHandler):
    def __init__(self, p=0.1, key_map_in=('prompt -> prompt',), key_map_out=('prompt -> prompt',)):
        super().__init__(key_map_in, key_map_out)
        self.p = p

    def handle(self, prompt):
        if isinstance(prompt, str):
            if random.random()<self.p:
                prompt = ''
        else:
            if random.random()<self.p:
                prompt['caption'] = ''
        return {'prompt':prompt}

    def __repr__(self):
        return f'TagEraseHandler(p={self.p})'


class TemplateFillHandler(DataHandler):
    def __init__(self, word_names: Dict[str, str], key_map_in=('prompt -> prompt',), key_map_out=('prompt -> prompt',)):
        super().__init__(key_map_in, key_map_out)
        self.word_names = word_names

    def handle(self, prompt):
        template, caption = prompt['template'], prompt['caption']

        keys_need = {i[1] for i in Formatter().parse(template) if i[1] is not None}
        fill_dict = {k: v for k, v in self.word_names.items() if k in keys_need}

        if (caption is not None) and ('caption' in keys_need):
            fill_dict.update(caption=fill_dict.get('caption', None) or caption)

        # skip keys that not provide
        for k in keys_need:
            if k not in fill_dict:
                fill_dict[k] = ''

        # replace None value with ''
        fill_dict = {k:(v or '') for k, v in fill_dict.items()}
        return {'prompt':template.format(**fill_dict)}

    def __repr__(self):
        return f'TemplateFill(\nword_names={self.word_names}\n)'

class TokenizeHandler(DataHandler):
    def __init__(self, encoder_attention_mask=False, key_map_in=('prompt -> prompt',), key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        self.encoder_attention_mask = encoder_attention_mask

        register_model_callback(self.acquire_tokenizer)

    def acquire_tokenizer(self, model_wrapper):
        self.tokenizer = model_wrapper.tokenizer

    def handle(self, prompt):
        token_info = self.tokenizer(prompt, truncation=True, padding="max_length", return_tensors="pt",
                                max_length=self.tokenizer.model_max_length*self.tokenizer.N_repeats)
        tokens = token_info.input_ids.squeeze()
        data = {'prompt':tokens}
        if self.encoder_attention_mask and 'attention_mask' in token_info:
            data['attn_mask'] = token_info.attention_mask.squeeze()
        if 'position_ids' in token_info:
            data['position_ids'] = token_info.position_ids.squeeze()

        return data

    def __repr__(self):
        return f'TokenizeHandler(\nencoder_attention_mask={self.encoder_attention_mask}, tokenizer={self.tokenizer}\n)'