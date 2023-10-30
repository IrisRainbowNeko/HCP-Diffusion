import torch
from torch.utils.data.distributed import DistributedSampler
from typing import Iterator
import platform
import math

class DistributedCycleSampler(DistributedSampler):
    _cycle = 1000

    def __iter__(self) -> Iterator:
        def _iter():
            while True:
                if self.shuffle:
                    # deterministically shuffle based on epoch and seed
                    g = torch.Generator()
                    g.manual_seed(self.seed + self.epoch)
                    indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
                else:
                    indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

                if not self.drop_last:
                    # add extra samples to make it evenly divisible
                    padding_size = self.total_size - len(indices)
                    if padding_size <= len(indices):
                        indices += indices[:padding_size]
                    else:
                        indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
                else:
                    # remove tail of data to make it evenly divisible.
                    indices = indices[:self.total_size]
                assert len(indices) == self.total_size

                # subsample
                indices = indices[self.rank:self.total_size:self.num_replicas]
                assert len(indices) == self.num_samples

                for idx in indices:
                    yield idx
                self.epoch+=1

                if self.epoch>=self._cycle:
                    break

        return _iter()

    def __len__(self) -> int:
        return self.num_samples #*self._cycle

def get_sampler():
    # Fix DataLoader frequently reload bugs in windows
    if platform.system().lower() == 'windows':
        return DistributedCycleSampler
    else:
        return DistributedSampler