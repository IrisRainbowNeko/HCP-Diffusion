import time
import warnings
from omegaconf import OmegaConf

def times(a,b):
    warnings.warn(f"${{times:{a},{b}}} is deprecated and will be removed in the future. Please use ${{hcp.eval:{a}*{b}}} instead.", DeprecationWarning)
    return a*b

OmegaConf.register_new_resolver("times", times)

OmegaConf.register_new_resolver("hcp.eval", lambda exp: eval(exp))
OmegaConf.register_new_resolver("hcp.time", lambda format="%Y-%m-%d-%H-%M-%S": time.strftime(format))