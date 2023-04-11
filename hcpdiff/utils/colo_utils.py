import torch
from typing import Any

import colossalai
from colossalai.utils import get_current_device
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.optimizer.zero_optimizer import ZeroOptimizer

class GeminiAdamOptimizerP(ZeroOptimizer):

    def __init__(self, model: torch.nn.Module, parameters, **defaults: Any) -> None:
        optimizer = HybridAdam(parameters, **defaults)
        super().__init__(optimizer, model, **defaults)

# Gemini + ZeRO DDP
def gemini_zero_dpp(model: torch.nn.Module, placememt_policy: str = "auto"):
    from colossalai.nn.parallel import GeminiDDP

    model = GeminiDDP(model,
                      device=get_current_device(),
                      placement_policy=placememt_policy,
                      pin_memory=True,
                      search_range_mb=64)
    return model