"""
pair_dataset.py
====================
    :Name:        text-image pair dataset
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from typing import Callable, Iterable
from .bucket import BaseBucket
import os.path

import torch
import cv2
from .pair_dataset import TextImagePairDataset
from hcpdiff.utils.utils import get_file_name
from torchvision import transforms

class TextImageCondPairDataset(TextImagePairDataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, prompt_template: str, tokenizer, tokenizer_repeats: int = 1, caption_file: str = None,
                 att_mask: str = None, att_mask_encode: bool = False, bg_color: Iterable[int] = (255, 255, 255),
                 image_transforms: Callable = None, tag_transforms: Callable = None, bucket: BaseBucket = None, return_path: bool = False,
                 cond_dir: str = None):
        super().__init__(prompt_template, tokenizer, tokenizer_repeats, caption_file, att_mask, att_mask_encode, bg_color, image_transforms,
                         tag_transforms, bucket, return_path)
        self.cond_dir = cond_dir
        self.cond_transform = transforms.ToTensor()

    def load_data(self, path, size):
        img_name = os.path.basename(path)
        image = self.load_image(path)
        img_cond = self.load_image(os.path.join(self.cond_dir, img_name))
        att_mask = self.get_att_map(get_file_name(img_name))
        if att_mask is None:
            data = self.bucket.crop_resize({"img": image, "cond":img_cond}, size)
            image = self.image_transforms(data['img'])  # resize to bucket size
            img_cond = self.cond_transform(data['cond'])
            att_mask = torch.ones((size[1] // 8, size[0] // 8))
        else:
            data = self.bucket.crop_resize({"img": image, "mask": att_mask, "cond":img_cond}, size)
            image = self.image_transforms(data['img'])
            img_cond = self.cond_transform(data['cond'])
            att_mask = torch.tensor(cv2.resize(att_mask, (size[0] // 8, size[1] // 8), interpolation=cv2.INTER_LINEAR))
        return {'img': image, 'mask': att_mask, "cond":img_cond}
