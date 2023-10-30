"""
pair_dataset.py
====================
    :Name:        text-image pair dataset
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import cv2
import torch

from .pair_dataset import TextImagePairDataset

class TextImageCondPairDataset(TextImagePairDataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def load_data(self, path, data_source, size):
        image_dict = data_source.load_image(path)
        image = image_dict['image']
        att_mask = image_dict.get('att_mask', None)
        img_cond = image_dict.get('cond', None)
        if img_cond is None:
            raise FileNotFoundError(f'{self.__class__} need the condition images!')

        if att_mask is None:
            data, crop_coord = self.bucket.crop_resize({"img":image, "cond":img_cond}, size)
            image = data_source.procees_image(data['img'])  # resize to bucket size
            img_cond = data_source.cond_transform(data['cond'])
            att_mask = torch.ones((size[1]//8, size[0]//8))
        else:
            data, crop_coord = self.bucket.crop_resize({"img":image, "mask":att_mask, "cond":img_cond}, size)
            image = data_source.procees_image(data['img'])
            img_cond = data_source.cond_transform(data['cond'])
            att_mask = torch.tensor(cv2.resize(data['mask'], (size[0]//8, size[1]//8), interpolation=cv2.INTER_LINEAR))
        return {'img':image, 'mask':att_mask, 'plugin_input':{"cond":img_cond}}
