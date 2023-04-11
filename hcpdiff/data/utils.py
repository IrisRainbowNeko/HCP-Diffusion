from PIL import Image
import torch
import cv2

from torchvision import transforms as T
from torchvision.transforms import functional as F


class DualRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        crop_params = T.RandomCrop.get_params(img['img'], self.size)
        img['img'] = F.crop(img['img'], *crop_params)
        if "mask" in img:
            img['mask'] = F.crop(img['mask'], *crop_params)
        return img

def resize_crop_fix(img, target_size):
    w,h=img['img'].size
    if w==target_size[0] and h==target_size[1]:
        return img

    ratio_img=w/h
    if ratio_img > target_size[0]/target_size[1]:
        new_size = (int(ratio_img*target_size[1]), target_size[1])
        img['img'] = img['img'].resize(new_size, (Image.ANTIALIAS if h>target_size[1] else Image.BICUBIC))
    else:
        new_size = (target_size[0], int(target_size[0]/ratio_img))
        img['img'] = img['img'].resize(new_size, (Image.ANTIALIAS if w>target_size[0] else Image.BICUBIC))
    if "mask" in img:
        img['mask'] = cv2.resize(img['mask'], new_size, interpolation=cv2.INTER_CUBIC)

    return DualRandomCrop(target_size[::-1])(img)

def collate_fn_ft(batch):
    imgs, att_mask, sn_list, sp_list = [], [], [], []
    for img, target in batch:
        imgs.append(img[0])
        att_mask.append(img[1])
        if len(target.shape)==2:
            sn_list.append(target[0])
            sp_list.append(target[1])
        else:
            sp_list.append(target)
    sn_list += sp_list
    return torch.stack(imgs), torch.stack(att_mask).unsqueeze(1), torch.stack(sn_list)