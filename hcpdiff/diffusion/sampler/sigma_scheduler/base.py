from typing import Union, Tuple

import torch

class SigmaScheduler:

    def get_sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        '''
        :param t: 0-1, rate of time step
        '''
        raise NotImplementedError

    def sample_sigma(self, min_rate=0.0, max_rate=1.0, shape=(1,)) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
