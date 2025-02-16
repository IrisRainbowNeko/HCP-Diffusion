"""
pair_dataset.py
====================
    :Name:        text-image pair dataset
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from rainbowneko.train.data import CacheableDataset

class TextImagePairDataset(CacheableDataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    pass
