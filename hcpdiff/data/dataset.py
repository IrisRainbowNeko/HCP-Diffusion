"""
pair_dataset.py
====================
    :Name:        text-image pair dataset
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from typing import Union, Dict

from rainbowneko.data import CacheableDataset, BaseDataset, BaseBucket, DataSource, DataHandler, DataCache

def TextImagePairDataset(bucket: BaseBucket = None, source: Dict[str, DataSource] = None, handler: DataHandler = None,
                         batch_handler: DataHandler = None, cache: DataCache = None, **kwargs) -> Union[CacheableDataset, BaseDataset]:
    if cache is None:
        return BaseDataset(bucket=bucket, source=source, handler=handler, batch_handler=batch_handler, **kwargs)
    else:
        return CacheableDataset(bucket=bucket, source=source, handler=handler, batch_handler=batch_handler, cache=cache, **kwargs)
