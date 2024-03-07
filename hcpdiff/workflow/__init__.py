from .base import BasicAction, from_memory, ExecAction, LoopAction
from .diffusion import InputFeederAction, PrepareDiffusionAction, MakeLatentAction, NoisePredAction, SampleAction, DiffusionStepAction, \
    X0PredAction, SeedAction, MakeTimestepsAction
from .text import TextEncodeAction, TextHookAction, AttnMultTextEncodeAction
from .vae import EncodeAction, DecodeAction
from .io import LoadModelsAction, SaveImageAction, BuildModelLoaderAction, LoadPartAction, LoadLoraAction, LoadPluginAction, LoadImageAction, \
    FeedInputAction
from .utils import LatentResizeAction, ImageResizeAction, FeedtoCNetAction
from .model import VaeOptimizeAction, BuildOffloadAction, XformersEnableAction, StartTextEncode, StartDiffusion, EndTextEncode, EndDiffusion, \
    BuildPluginAction

try:
    from .fast import SFastCompileAction
except:
    print('stable fast not installed.')

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("hcp.from_memory", lambda mem_name:OmegaConf.create({
    '_target_':'hcpdiff.workflow.from_memory',
    'mem_name':mem_name,
}))
