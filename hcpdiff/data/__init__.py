from .pair_dataset import TextImagePairDataset
from .cond_dataset import TextImageCondPairDataset
from .crop_info_dataset import CropInfoPairDataset
from .bucket import BaseBucket, FixedBucket, RatioBucket
from .utils import CycleData
from .caption_loader import JsonCaptionLoader, TXTCaptionLoader

class DataGroup:
    def __init__(self, loader_list, loss_weights):
        self.loader_list = loader_list
        self.loss_weights = loss_weights

    def __iter__(self):
        self.data_iter_list = [iter(CycleData(loader)) for loader in self.loader_list]
        return self

    def __next__(self):
        return [next(data_iter) for data_iter in self.data_iter_list]

    def __len__(self):
        return len(self.loader_list)

    def get_dataset(self, idx):
        return self.loader_list[idx].dataset

    def get_loss_weights(self, idx):
        return self.loss_weights[idx]