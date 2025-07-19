from io import BytesIO
from pathlib import Path
from typing import Dict, Any

import lmdb
import torch
from hcpdiff.models.wrapper import SD15Wrapper
from rainbowneko import _share
from rainbowneko.data import DataCache, CacheableDataset
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
        cached_data = self.load_latent(data['id'])
        data['image'] = cached_data['latent']
        data['coord'] = cached_data['coord']
        return data
    
    def on_finish(self, index, data):
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

    def build(self, dataset: CacheableDataset, model: SD15Wrapper, all_gather):
        if (self.pre_build and Path(self.pre_build).exists()) or len(self.cache)>0:
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
                        image = data['image'].to(device=_share.device, dtype=vae.dtype)
                        latents = model.vae.encode(image).latent_dist.sample()
                        if shift_factor := getattr(vae.config, 'shift_factor', None) is not None:
                            latents = (latents-shift_factor)*vae.config.scaling_factor
                        else:
                            latents = latents*vae.config.scaling_factor
                        latents = latents.cpu()

                        for img_id, latent, coord in zip(data['id'], latents, data['coord']):
                            data_cache = {'latent': latent, 'coord': coord}

                            byte_stream = BytesIO()
                            torch.save(data_cache, byte_stream)
                            txn.put(str(img_id).encode(), byte_stream.getvalue())
                            if not self.lazy:
                                self.cache[img_id] = data_cache
                env.close()
            else:
                for data in tqdm(loader):
                    img_id = data['id']
                    image = data['image'].to(device=_share.device, dtype=vae.dtype)
                    latents = model.vae.encode(image).latent_dist.sample()
                    if shift_factor := getattr(vae.config, 'shift_factor', None) is not None:
                        latents = (latents-shift_factor)*vae.config.scaling_factor
                    else:
                        latents = latents*vae.config.scaling_factor
                    latents = latents.cpu()
                    for img_id, latent, coord in zip(data['id'], latents, data['coord']):
                        self.cache[img_id] = {'latent': latent, 'coord': coord}

        model.vae.to('cpu')
        #model.vae = None
        torch.cuda.empty_cache()

        cache_all = all_gather(self.cache)
        for cache in cache_all:
            self.cache.update(cache)
