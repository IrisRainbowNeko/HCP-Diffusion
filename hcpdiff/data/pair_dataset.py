"""
pair_dataset.py
====================
    :Name:        text-image pair dataset
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from typing import Tuple, Callable, Iterable
import os.path

import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from .bucket import BaseBucket
from hcpdiff.utils.caption_tools import *
from hcpdiff.utils.utils import _default, get_file_name, get_file_ext
from hcpdiff.utils.img_size_tool import types_support
from tqdm.auto import tqdm

import json
import yaml


class TextImagePairDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, prompt_template: str, tokenizer, tokenizer_repeats: int = 1, caption_file: str = None,
                 att_mask: str = None, att_mask_encode: bool = False, bg_color: Iterable[int] = (255, 255, 255),
                 image_transforms: Callable = None, tag_transforms: Callable = None, bucket: BaseBucket = None, return_path: bool = False):
        self.return_path = return_path

        self.tokenizer = tokenizer
        self.tokenizer_repeats = tokenizer_repeats

        self.bucket: BaseBucket = bucket
        self.caption_dict = self.load_captions(caption_file)
        self.att_mask_path = {} if att_mask is None else \
            {get_file_name(file): os.path.join(att_mask, file) for file in os.listdir(att_mask) if
             get_file_ext(file) in types_support}
        self.att_mask_encode = att_mask_encode

        self.prompt_template = self.load_template(prompt_template)

        self.image_transforms = _default(image_transforms, image_transforms)
        self.tag_transforms = _default(tag_transforms, tag_transforms)
        self.bg_color = tuple(bg_color)

        self.latents = None  # Cache latents for faster training. Works only without image argumentations.

    def load_image(self, path):
        image = Image.open(path)
        if image.mode == 'RGBA':
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas
        return image.convert("RGB")

    def load_captions(self, caption_file):
        if caption_file is None:
            return dict()
        elif caption_file.endswith('.json'):
            with open(caption_file, 'r', encoding='utf-8') as f:
                return json.loads(f.read())
        elif caption_file.endswith('.yaml'):
            with open(caption_file, 'r', encoding='utf-8') as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            return dict()

    def load_template(self, template_file):
        with open(template_file, 'r', encoding='utf-8') as f:
            return f.read().strip().split('\n')

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
                data['img'] = (latents * 0.18215).cpu()
                self.latents[img_name] = data

    def get_att_map(self, img_name):
        if img_name not in self.att_mask_path:
            return None
        att_mask = Image.open(self.att_mask_path[img_name]).convert("L")
        np_mask = np.array(att_mask).astype(float)
        np_mask[np_mask <= 127 + 0.1] = (np_mask[np_mask <= 127 + 0.1] / 127.)
        np_mask[np_mask > 127] = ((np_mask[np_mask > 127] - 127) / 128.) * 4 + 1
        return np_mask

    def load_data(self, path, size):
        img_name = os.path.basename(path)
        image = self.load_image(path)
        att_mask = self.get_att_map(get_file_name(img_name))
        if att_mask is None:
            data = self.bucket.crop_resize({"img": image}, size)
            image = self.image_transforms(data['img'])  # resize to bucket size
            att_mask = torch.ones((size[1] // 8, size[0] // 8))
        else:
            data = self.bucket.crop_resize({"img": image, "mask": att_mask}, size)
            image = self.image_transforms(data['img'])
            att_mask = torch.tensor(cv2.resize(att_mask, (size[0] // 8, size[1] // 8), interpolation=cv2.INTER_LINEAR))
        return {'img': image, 'mask': att_mask}

    def __len__(self):
        return len(self.bucket)

    def __getitem__(self, index):
        path, size = self.bucket[index]
        img_name = os.path.basename(path)

        if self.latents is None:
            data = self.load_data(path, size)
        else:
            data = self.latents[img_name]

        caption_ist = self.caption_dict[img_name] if img_name in self.caption_dict else None
        prompt_ist = self.tag_transforms({'prompt': random.choice(self.prompt_template), 'caption': caption_ist})['prompt']

        # tokenize Sp or (Sn, Sp)
        prompt_ids = self.tokenizer(
            prompt_ist, truncation=True, padding="max_length", return_tensors="pt",
            max_length=self.tokenizer.model_max_length * self.tokenizer_repeats).input_ids.squeeze()

        if self.return_path:
            return data, prompt_ids, path
        else:
            return data, prompt_ids
