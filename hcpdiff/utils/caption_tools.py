"""
caption_tools.py
====================
    :Name:        process prompts
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import random
from string import Formatter
from typing import List, Dict, Union

import numpy as np


class TagShuffle:
    def __call__(self, data):
        if 'caption' in data:
            text = data['caption']
            if text is not None:
                tags = text.split(',')
                random.shuffle(tags)
                data['caption'] = ','.join(tags)
            return data
        else:
            for i, item in enumerate(data['prompt']):
                tags = item.split(',')
                random.shuffle(tags)
                data['prompt'][i] = ','.join(tags)
            return data

    def __repr__(self):
        return 'TagShuffle()'


class TagDropout:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, data):
        if 'caption' in data:
            text = data['caption']
            if text is not None:
                tags = np.array(text.split(','))
                data['caption'] = ','.join(tags[np.random.random(len(tags)) > self.p])
            return data
        else:
            for i, item in enumerate(data['prompt']):
                tags = item.split(',')
                data['prompt'][i] = ','.join(tags[np.random.random(len(tags)) > self.p])
            return data

    def __repr__(self):
        return f'TagDropout(p={self.p})'


class TemplateFill:
    def __init__(self, word_names: Dict[str, Union[str, List[str]]]):
        self.word_names = word_names
        self.DA_names = {k: v for k, v in word_names.items() if not isinstance(v, str)}
        self.dream_artist = len(self.DA_names) > 0

    def __call__(self, data):
        template, caption = data['prompt'], data['caption']

        keys_need = {i[1] for i in Formatter().parse(template) if i[1] is not None}
        fill_dict = {k: v for k, v in self.word_names.items() if k in keys_need}

        if (caption is not None) and ('caption' in keys_need):
            if self.dream_artist:
                cap_fill = fill_dict.get('caption', [None, None])
                fill_dict.update(caption=[cap_fill[0] or caption, cap_fill[1] or caption])
            else:
                fill_dict.update(caption=fill_dict.get('caption', None) or caption)

        # skip keys that not provide
        for k in keys_need:
            if k not in fill_dict:
                fill_dict[k] = ''

        if self.dream_artist:
            fill_dict_pos = {k: ((v if isinstance(v, str) else v[0]) or '') for k, v in fill_dict.items()}
            fill_dict_neg = {k: ((v if isinstance(v, str) else v[1]) or '') for k, v in fill_dict.items()}
            return {'prompt':[template.format(**fill_dict_neg), template.format(**fill_dict_pos)]}
        else:
            # replace None value with ''
            fill_dict = {k:(v or '') for k, v in fill_dict.items()}
            return {'prompt':[template.format(**fill_dict)]}

    def __repr__(self):
        return f'TemplateFill(\nword_names={self.word_names}\n)'
