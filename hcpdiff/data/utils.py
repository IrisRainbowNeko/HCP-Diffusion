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
        if "cond" in img:
            img['cond'] = F.crop(img['cond'], *crop_params)
        return img

def resize_crop_fix(img, target_size):
    w,h=img['img'].size
    if w==target_size[0] and h==target_size[1]:
        return img

    ratio_img=w/h
    if ratio_img > target_size[0]/target_size[1]:
        new_size = (int(ratio_img*target_size[1]), target_size[1])
        interp_type = Image.ANTIALIAS if h>target_size[1] else Image.BICUBIC
    else:
        new_size = (target_size[0], int(target_size[0]/ratio_img))
        interp_type = Image.ANTIALIAS if w>target_size[0] else Image.BICUBIC
    img['img'] = img['img'].resize(new_size, interp_type)
    if "mask" in img:
        img['mask'] = cv2.resize(img['mask'], new_size, interpolation=cv2.INTER_CUBIC)
    if "cond" in img:
        img['cond'] = img['cond'].resize(new_size, interp_type)

    return DualRandomCrop(target_size[::-1])(img)

def collate_fn_ft(batch):
    datas, sn_list, sp_list = {'img':[]}, [], []

    data0 = batch[0]
    if 'mask' in data0:
        datas['mask']=[]
    if 'cond' in data0:
        datas['cond']=[]

    for data in batch:
        datas['img'].append(data['img'])
        datas['mask'].append(data['mask'])
        if 'cond' in data:
            datas['cond'].append(data['cond'])

        target = data['prompt']
        if len(target.shape)==2:
            sn_list.append(target[0])
            sp_list.append(target[1])
        else:
            sp_list.append(target)
    sn_list += sp_list

    datas['img'] = torch.stack(datas['img'])
    datas['mask'] = torch.stack(datas['mask']).unsqueeze(1)
    if 'cond' in data0:
        datas['cond'] = torch.stack(datas['cond'])
    datas['prompt'] = torch.stack(sn_list)

    return datas

class CycleData():
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.epoch = 0
        def cycle():
            while True:
                self.data_loader.dataset.bucket.rest(self.epoch)
                for data in self.data_loader:
                    yield data
                self.epoch+=1
        return cycle()