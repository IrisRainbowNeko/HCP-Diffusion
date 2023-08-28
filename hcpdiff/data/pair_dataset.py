"""
pair_dataset.py
====================
    :Name:        text-image pair dataset
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import os.path
from argparse import Namespace

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from hcpdiff.utils.caption_tools import *
from hcpdiff.utils.img_size_tool import types_support
from hcpdiff.utils.utils import get_file_name, get_file_ext
from .bucket import BaseBucket
from .caption_loader import BaseCaptionLoader, auto_caption_loader

class TextImagePairDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, tokenizer, tokenizer_repeats: int = 1, att_mask_encode: bool = False,
                 bucket: BaseBucket = None, source: Dict = None, return_path: bool = False, **kwargs):
        self.return_path = return_path

        self.tokenizer = tokenizer
        self.tokenizer_repeats = tokenizer_repeats
        self.bucket: BaseBucket = bucket
        self.att_mask_encode = att_mask_encode
        self.load_source(source)
        self.latents = None  # Cache latents for faster training. Works only without image argumentations.

    def load_source(self, source: Dict):
        self.source_dict = {}
        for data_source in source.values():
            source_metas = Namespace()
            source_metas.caption_dict = self.load_captions(data_source.caption_file)
            source_metas.att_mask_path = {} if data_source.att_mask is None else \
                {get_file_name(file):os.path.join(data_source.att_mask, file)
                    for file in os.listdir(data_source.att_mask) if get_file_ext(file) in types_support}
            source_metas.prompt_template = self.load_template(data_source.prompt_template)
            source_metas.image_transforms = data_source.image_transforms
            source_metas.tag_transforms = data_source.tag_transforms
            source_metas.bg_color = tuple(data_source.bg_color)
            source_metas.repeat = getattr(data_source, 'repeat', 1)

            self.source_dict[os.path.dirname(data_source.img_root+'/')] = source_metas

    def load_image(self, path):
        image = Image.open(path)
        if image.mode == 'RGBA':
            img_root = os.path.dirname(path)
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.source_dict[img_root].bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas
        return image.convert("RGB")

    def load_captions(self, caption_file: Union[str, BaseCaptionLoader]):
        if caption_file is None:
            return {}
        elif isinstance(caption_file, str):
            return auto_caption_loader(caption_file).load()
        else:
            return caption_file.load()

    def load_template(self, template_file):
        with open(template_file, 'r', encoding='utf-8') as f:
            return f.read().strip().split('\n')

    def load_data(self, path, size):
        img_root, img_name = os.path.split(path)
        image = self.load_image(path)
        im_w, im_h = image.size
        att_mask = self.get_att_map(img_root, get_file_name(img_name))
        if att_mask is None:
            data, crop_coord = self.bucket.crop_resize({"img":image}, size)
            image = self.source_dict[img_root].image_transforms(data['img'])  # resize to bucket size
            att_mask = torch.ones((size[1]//8, size[0]//8))
        else:
            data, crop_coord = self.bucket.crop_resize({"img":image, "mask":att_mask}, size)
            image = self.source_dict[img_root].image_transforms(data['img'])
            att_mask = torch.tensor(cv2.resize(data['mask'], (size[0]//8, size[1]//8), interpolation=cv2.INTER_LINEAR))
        return {'img':image, 'mask':att_mask}

    @torch.no_grad()
    def cache_latents(self, vae, weight_dtype, device, show_prog=True):
        self.latents = {}
        self.bucket.rest(0)

        for path, size in tqdm(self.bucket, disable=not show_prog):
            img_name = os.path.basename(path)
            if img_name not in self.latents:
                data = self.load_data(path, size)
                image = data['img'].unsqueeze(0).to(device, dtype=weight_dtype)
                latents = vae.encode(image).latent_dist.sample().squeeze(0)
                data['img'] = (latents*vae.scaling_factor).cpu()
                self.latents[img_name] = data

    def get_att_map(self, img_root, img_name):
        if img_name not in self.source_dict[img_root].att_mask_path:
            return None
        att_mask = Image.open(self.source_dict[img_root].att_mask_path[img_name]).convert("L")
        np_mask = np.array(att_mask).astype(float)
        np_mask[np_mask<=127+0.1] = (np_mask[np_mask<=127+0.1]/127.)
        np_mask[np_mask>127] = ((np_mask[np_mask>127]-127)/128.)*4+1
        return np_mask

    def __len__(self):
        return len(self.bucket)

    def __getitem__(self, index):
        path, size = self.bucket[index]
        img_root, img_name = os.path.split(path)

        if self.latents is None:
            data = self.load_data(path, size)
        else:
            data = self.latents[img_name].copy()

        caption_ist = self.source_dict[img_root].caption_dict.get(img_name, None)
        prompt_template = random.choice(self.source_dict[img_root].prompt_template)
        prompt_ist = self.source_dict[img_root].tag_transforms({'prompt':prompt_template, 'caption':caption_ist})['prompt']

        # tokenize Sp or (Sn, Sp)
        prompt_ids = self.tokenizer(prompt_ist, truncation=True, padding="max_length", return_tensors="pt",
                                    max_length=self.tokenizer.model_max_length*self.tokenizer_repeats).input_ids.squeeze()
        data['prompt'] = prompt_ids

        if self.return_path:
            return data, path
        else:
            return data

    @staticmethod
    def collate_fn(batch):
        '''
        batch: [{img:tensor, prompt:str, ..., plugin_input:{...}},{}]
        '''
        if 'plugin_input' in batch[0]:
            plugin_input = {k:[] for k in batch[0]['plugin_input'].keys()}

        datas = {k:[] for k in batch[0].keys() if k != 'plugin_input' and k != 'prompt'}
        sn_list, sp_list = [], []

        for data in batch:
            if 'plugin_input' in data:
                for k, v in data.pop('plugin_input').items():
                    plugin_input[k].append(v)

            prompt = data.pop('prompt')
            if len(prompt.shape) == 2:
                sn_list.append(prompt[0])
                sp_list.append(prompt[1])
            else:
                sp_list.append(prompt)

            for k, v in data.items():
                datas[k].append(v)

        for k, v in datas.items():
            datas[k] = torch.stack(v)
            if k == 'mask':
                datas[k] = datas[k].unsqueeze(1)

        sn_list += sp_list
        datas['prompt'] = torch.stack(sn_list)
        if 'plugin_input' in batch[0]:
            datas['plugin_input'] = {k:torch.stack(v) for k, v in plugin_input.items()}

        return datas
