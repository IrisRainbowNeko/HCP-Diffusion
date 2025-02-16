from io import BytesIO
from pathlib import Path
from typing import Dict, Any

import lmdb
import torch
from hcpdiff.models.wrapper import StableDiffusionWrapper
from rainbowneko import _share
from rainbowneko.train.data import DataCache, CacheableDataset
from rainbowneko.utils import Path_Like
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

class VaeCache(DataCache):
    def __init__(self, pre_build: Path_Like = None, lazy=False, bs=1):
        super().__init__(pre_build)
        self.lazy = lazy
        self.bs = bs

    def load_latent(self, id):
        if self.lazy:
            with self.env.begin() as txn:
                byte_tensor = txn.get(str(id).encode())
                return torch.load(BytesIO(byte_tensor))
        else:
            return self.cache[id]

    def before_handler(self, index: int, data: Dict[str, Any]):
        data['image'] = self.load_latent(data['id'])
        return data

    def load(self, path):
        if self.lazy:
            self.env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
            return {}
        elif len(self.cache)>0:
            return self.cache
        else:
            env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin() as txn:
                cache = {k.decode():torch.load(BytesIO(v)) for k, v in txn.cursor()}
            env.close()
            return cache

    def build(self, dataset: CacheableDataset, model: StableDiffusionWrapper):
        if Path(self.pre_build).exists() or len(self.cache)>0:
            model.vae = None
            return

        vae = model.vae.to(_share.device)
        with dataset.disable_cache():
            dataset.bucket.rest(0)

            loader = DataLoader(
                dataset,
                batch_size=self.bs,
                num_workers=0,
                sampler=DistributedSampler(dataset, num_replicas=_share.world_size, rank=_share.local_rank, shuffle=False),
                collate_fn=dataset.collate_fn,
                drop_last=False,
            )

            if self.pre_build:
                Path(self.pre_build).parent.mkdir(parents=True, exist_ok=True)
                env = lmdb.open(self.pre_build, map_size=1099511627776)
                with env.begin(write=True) as txn:
                    for data in tqdm(loader):
                        img_id = data['id']
                        image = data['image'].to(device=_share.device, dtype=vae.dtype)
                        latents = model.vae.encode(image).latent_dist.sample()
                        latents = (latents*vae.config.scaling_factor).cpu()
                        byte_stream = BytesIO()
                        torch.save(latents, byte_stream)
                        txn.put(str(img_id).encode(), byte_stream.getvalue())
                        if not self.lazy:
                            self.cache[img_id] = latents
                env.close()
            else:
                for data in tqdm(loader):
                    img_id = data['id']
                    image = data['image'].to(device=_share.device, dtype=vae.dtype)
                    latents = model.vae.encode(image).latent_dist.sample()
                    latents = (latents*vae.config.scaling_factor).cpu()
                    self.cache[img_id] = latents

        model.vae = None
