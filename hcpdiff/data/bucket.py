"""
bucket.py
====================
    :Name:        aspect ratio bucket with k-means
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import math
import os.path
import pickle
from typing import List, Tuple, Union, Any

import cv2
import numpy as np
from hcpdiff.utils.img_size_tool import types_support, get_image_size
from hcpdiff.utils.utils import get_file_ext
from loguru import logger
from sklearn.cluster import KMeans

from .utils import resize_crop_fix, pad_crop_fix

class BaseBucket:
    def __getitem__(self, idx):
        '''
        :return: (file name of image), (target image size)
        '''
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def build(self, bs: int, img_root_list: List[str]):
        raise NotImplementedError()

    def rest(self, epoch):
        pass

    def crop_resize(self, image, size, mask_interp=cv2.INTER_CUBIC) -> Tuple[Any, Tuple]:
        return image, (*size, 0, 0, *size)

class FixedBucket(BaseBucket):
    def __init__(self, target_size: Union[Tuple[int, int], int] = 512, **kwargs):
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size

    def build(self, bs: int, img_root_list: List[str]):
        self.img_root_list = img_root_list
        self.file_names = []
        for img_root, repeat in img_root_list:
            imgs = [os.path.join(img_root, x) for x in os.listdir(img_root) if get_file_ext(x) in types_support]
            self.file_names.extend(imgs*repeat)

    def crop_resize(self, image, size, mask_interp=cv2.INTER_CUBIC):
        return resize_crop_fix(image, size, mask_interp=mask_interp)

    def __getitem__(self, idx) -> Tuple[str, Tuple[int, int]]:
        return self.file_names[idx], self.target_size

    def __len__(self):
        return len(self.file_names)

class RatioBucket(BaseBucket):
    def __init__(self, taget_area: int = 640*640, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None):
        self.taget_area = taget_area
        self.step_size = step_size
        self.num_bucket = num_bucket
        self.pre_build_bucket = pre_build_bucket

    def load_bucket(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.buckets = data['buckets']
        self.size_buckets = data['size_buckets']
        self.file_names = data['file_names']
        self.idx_bucket_map = data['idx_bucket_map']
        self.data_len = data['data_len']

    def save_bucket(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'buckets':self.buckets,
                'size_buckets':self.size_buckets,
                'idx_bucket_map':self.idx_bucket_map,
                'file_names':self.file_names,
                'data_len':self.data_len,
            }, f)

    def build_buckets_from_ratios(self):
        logger.info('build buckets from ratios')
        size_low = int(math.sqrt(self.taget_area/self.ratio_max))
        size_high = int(self.ratio_max*size_low)

        # SD需要边长是8的倍数
        size_low = (size_low//self.step_size)*self.step_size
        size_high = (size_high//self.step_size)*self.step_size

        data = []
        for w in range(size_low, size_high+1, self.step_size):
            for h in range(size_low, size_high+1, self.step_size):
                data.append([w*h, np.log2(w/h), w, h])  # 对比例取对数，更符合人感知，宽高相反的可以对称分布。
        data = np.array(data)

        error_area = np.abs(data[:, 0]-self.taget_area)
        data_use = data[np.argsort(error_area)[:self.num_bucket*3], :]  # 取最小的num_bucket*3个

        # 聚类，选出指定个数的bucket
        kmeans = KMeans(n_clusters=self.num_bucket, random_state=3407).fit(data_use[:, 1].reshape(-1, 1))
        labels = kmeans.labels_
        self.buckets = []  # [bucket_id:[file_idx,...]]
        self.ratios_log = []
        self.size_buckets = []
        for i in range(self.num_bucket):
            map_idx = np.where(labels == i)[0]
            m_idx = map_idx[np.argmin(np.abs(data_use[labels == i, 1]-np.median(data_use[labels == i, 1])))]
            # self.buckets[wh_hash(*data_use[m_idx, 2:])]=[]
            self.buckets.append([])
            self.ratios_log.append(data_use[m_idx, 1])
            self.size_buckets.append(data_use[m_idx, 2:].astype(int))
        self.ratios_log = np.array(self.ratios_log)
        self.size_buckets = np.array(self.size_buckets)

        # fill buckets with images w,h
        self.idx_bucket_map = np.empty(len(self.file_names), dtype=int)
        for i, file in enumerate(self.file_names):
            w, h = get_image_size(file)
            bucket_id = np.abs(self.ratios_log-np.log2(w/h)).argmin()
            self.buckets[bucket_id].append(i)
            self.idx_bucket_map[i] = bucket_id
        logger.info('buckets info: '+', '.join(f'size:{self.size_buckets[i]}, num:{len(b)}' for i, b in enumerate(self.buckets)))

    def build_buckets_from_images(self):
        logger.info('build buckets from images')
        ratio_list = []
        for i, file in enumerate(self.file_names):
            w, h = get_image_size(file)
            ratio = np.log2(w/h)
            ratio_list.append(ratio)
        ratio_list = np.array(ratio_list)

        # 聚类，选出指定个数的bucket
        kmeans = KMeans(n_clusters=self.num_bucket, random_state=3407).fit(ratio_list.reshape(-1, 1))
        labels = kmeans.labels_
        self.ratios_log = kmeans.cluster_centers_.reshape(-1)

        ratios = 2**self.ratios_log
        h_all = np.sqrt(self.taget_area/ratios)
        w_all = h_all*ratios

        # SD需要边长是8的倍数
        h_all = (np.round(h_all/self.step_size)*self.step_size).astype(int)
        w_all = (np.round(w_all/self.step_size)*self.step_size).astype(int)
        self.size_buckets = list(zip(w_all, h_all))
        self.size_buckets = np.array(self.size_buckets)

        self.buckets = []  # [bucket_id:[file_idx,...]]
        self.idx_bucket_map = np.empty(len(self.file_names), dtype=int)
        for bidx in range(self.num_bucket):
            bnow = labels == bidx
            self.buckets.append(np.where(bnow)[0].tolist())
            self.idx_bucket_map[bnow] = bidx
        logger.info('buckets info: '+', '.join(f'size:{self.size_buckets[i]}, num:{len(b)}' for i, b in enumerate(self.buckets)))

    def build(self, bs: int, img_root_list: List[str]):
        '''
        :param bs: batch_size * n_gpus * accumulation_step
        :param img_root_list:
        '''
        self.img_root_list = img_root_list
        if self.pre_build_bucket and os.path.exists(self.pre_build_bucket):
            self.load_bucket(self.pre_build_bucket)
            return
        else:
            self.file_names = []
            for img_root, repeat in img_root_list:
                imgs = [os.path.join(img_root, x) for x in os.listdir(img_root) if get_file_ext(x) in types_support]
                self.file_names.extend(imgs*repeat)
        self._build()

        self.bs = bs
        rs = np.random.RandomState(42)
        # make len(bucket)%bs==0
        self.data_len = 0
        for bidx, bucket in enumerate(self.buckets):
            rest = len(bucket)%bs
            if rest>0:
                bucket.extend(rs.choice(bucket, bs-rest))
            self.data_len += len(bucket)
            self.buckets[bidx] = np.array(bucket)

        if self.pre_build_bucket:
            self.save_bucket(self.pre_build_bucket)

    def rest(self, epoch):
        rs = np.random.RandomState(42+epoch)
        bucket_list = [x.copy() for x in self.buckets]
        # shuffle inter bucket
        for x in bucket_list:
            rs.shuffle(x)

        # shuffle of batches
        bucket_list = np.hstack(bucket_list).reshape(-1, self.bs).astype(int)
        rs.shuffle(bucket_list)

        self.idx_bucket = bucket_list.reshape(-1)

    def crop_resize(self, image, size, mask_interp=cv2.INTER_CUBIC):
        return resize_crop_fix(image, size, mask_interp=mask_interp)

    def __getitem__(self, idx):
        file_idx = self.idx_bucket[idx]
        bucket_idx = self.idx_bucket_map[file_idx]
        return self.file_names[file_idx], self.size_buckets[bucket_idx]

    def __len__(self):
        return self.data_len

    @classmethod
    def from_ratios(cls, target_area: int = 640*640, step_size: int = 8, num_bucket: int = 10, ratio_max: float = 4,
                    pre_build_bucket: str = None, **kwargs):
        arb = cls(target_area, step_size, num_bucket, pre_build_bucket=pre_build_bucket)
        arb.ratio_max = ratio_max
        arb._build = arb.build_buckets_from_ratios
        return arb

    @classmethod
    def from_files(cls, target_area: int = 640*640, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None, **kwargs):
        arb = cls(target_area, step_size, num_bucket, pre_build_bucket=pre_build_bucket)
        arb._build = arb.build_buckets_from_images
        return arb

class SizeBucket(RatioBucket):
    def __init__(self, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None):
        super().__init__(step_size=step_size, num_bucket=num_bucket, pre_build_bucket=pre_build_bucket)

    def build_buckets_from_images(self):
        '''
        根据图像尺寸聚类，不会resize图像，只有剪裁和填充操作。
        '''
        logger.info('build buckets from images size')
        size_list = []
        for i, file in enumerate(self.file_names):
            w, h = get_image_size(file)
            size_list.append([w, h])
        size_list = np.array(size_list)

        # 聚类，选出指定个数的bucket
        kmeans = KMeans(n_clusters=self.num_bucket, random_state=3407).fit(size_list)
        labels = kmeans.labels_
        size_buckets = kmeans.cluster_centers_

        # SD需要边长是8的倍数
        self.size_buckets = (np.round(size_buckets/self.step_size)*self.step_size).astype(int)

        self.buckets = []  # [bucket_id:[file_idx,...]]
        self.idx_bucket_map = np.empty(len(self.file_names), dtype=int)
        for bidx in range(self.num_bucket):
            bnow = labels == bidx
            self.buckets.append(np.where(bnow)[0].tolist())
            self.idx_bucket_map[bnow] = bidx
        logger.info('buckets info: '+', '.join(f'size:{self.size_buckets[i]}, num:{len(b)}' for i, b in enumerate(self.buckets)))

    def crop_resize(self, image, size):
        return pad_crop_fix(image, size)

    @classmethod
    def from_files(cls, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None, **kwargs):
        arb = cls(step_size, num_bucket, pre_build_bucket=pre_build_bucket)
        arb._build = arb.build_buckets_from_images
        return arb

class LongEdgeBucket(RatioBucket):
    def __init__(self, target_edge=640, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None):
        super().__init__(step_size=step_size, num_bucket=num_bucket, pre_build_bucket=pre_build_bucket)
        self.target_edge = target_edge

    def build_buckets_from_images(self):
        '''
        根据图像尺寸聚类，不会resize图像，只有剪裁和填充操作。
        '''
        logger.info('build buckets from images size')
        size_list = []
        for i, file in enumerate(self.file_names):
            w, h = get_image_size(file)
            scale = self.target_edge/max(w, h)
            size_list.append([round(w*scale), round(h*scale)])
        size_list = np.array(size_list)

        # 聚类，选出指定个数的bucket
        kmeans = KMeans(n_clusters=self.num_bucket, random_state=3407).fit(size_list)
        labels = kmeans.labels_
        size_buckets = kmeans.cluster_centers_

        # SD需要边长是8的倍数
        self.size_buckets = (np.round(size_buckets/self.step_size)*self.step_size).astype(int)

        self.buckets = []  # [bucket_id:[file_idx,...]]
        self.idx_bucket_map = np.empty(len(self.file_names), dtype=int)
        for bidx in range(self.num_bucket):
            bnow = labels == bidx
            self.buckets.append(np.where(bnow)[0].tolist())
            self.idx_bucket_map[bnow] = bidx
        logger.info('buckets info: '+', '.join(f'size:{self.size_buckets[i]}, num:{len(b)}' for i, b in enumerate(self.buckets)))

    def crop_resize(self, image, size):
        return pad_crop_fix(image, size)

    @classmethod
    def from_files(cls, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None, **kwargs):
        arb = cls(step_size, num_bucket, pre_build_bucket=pre_build_bucket)
        arb._build = arb.build_buckets_from_images
        return arb
