import time
import warnings
from omegaconf import OmegaConf
import torch
from .net_utils import dtype_dict

def times(a,b):
    warnings.warn(f"${{times:{a},{b}}} is deprecated and will be removed in the future. Please use ${{hcp.eval:{a}*{b}}} instead.", DeprecationWarning)
    return a*b

OmegaConf.register_new_resolver("times", times)

OmegaConf.register_new_resolver("hcp.eval", lambda exp: eval(exp))
OmegaConf.register_new_resolver("hcp.time", lambda format="%Y-%m-%d-%H-%M-%S": time.strftime(format))

OmegaConf.register_new_resolver("hcp.dtype", lambda dtype: dtype_dict.get(dtype, torch.float32))