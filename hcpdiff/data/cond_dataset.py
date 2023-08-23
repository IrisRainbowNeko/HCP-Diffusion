"""
pair_dataset.py
====================
    :Name:        text-image pair dataset
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from typing import Callable, Iterable, Dict
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

    def load_source(self, source: Dict):
        super(TextImageCondPairDataset, self).load_source(source)
        for data_source in source.values():
            self.source_dict[data_source.img_root].cond_dir = data_source.cond_dir
        self.cond_transform = transforms.ToTensor()

    def load_data(self, path, size):
        img_root, img_name = os.path.split(path)
        image = self.load_image(path)
        img_cond = self.load_image(os.path.join(self.source_dict[img_root].cond_dir, img_name))
        att_mask = self.get_att_map(img_root, get_file_name(img_name))
        if att_mask is None:
            data, crop_coord = self.bucket.crop_resize({"img": image, "cond":img_cond}, size)
            image = self.source_dict[img_root].image_transforms(data['img'])  # resize to bucket size
            img_cond = self.cond_transform(data['cond'])
            att_mask = torch.ones((size[1] // 8, size[0] // 8))
        else:
            data, crop_coord = self.bucket.crop_resize({"img": image, "mask": att_mask, "cond":img_cond}, size)
            image = self.source_dict[img_root].image_transforms(data['img'])
            img_cond = self.cond_transform(data['cond'])
            att_mask = torch.tensor(cv2.resize(data['mask'], (size[0] // 8, size[1] // 8), interpolation=cv2.INTER_LINEAR))
        return {'img': image, 'mask': att_mask, 'plugin_input':{"cond":img_cond}}
