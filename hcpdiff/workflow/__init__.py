from .base import BasicAction, MemoryMixin, from_memory, ExecAction, LoopAction
from .diffusion import InputFeederAction, PrepareDiffusionAction, MakeLatentAction, NoisePredAction, SampleAction, DiffusionStepAction, \
    X0PredAction, SeedAction, MakeTimestepsAction
from .text import TextEncodeAction, TextHookAction, AttnMultTextEncodeAction
from .vae import EncodeAction, DecodeAction
from .io import LoadModelsAction, SaveImageAction, BuildModelLoaderAction, LoadPartAction, LoadLoraAction, LoadPluginAction
from .utils import LatentResizeAction, ImageResizeAction
from .model import VaeOptimizeAction, BuildOffloadAction, XformersEnableAction, StartTextEncode, StartDiffusion, EndTextEncode, EndDiffusion

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("hcp.from_memory", lambda mem_name: OmegaConf.create({
    '_target_': 'hcpdiff.workflow.from_memory',
    'mem_name': mem_name,
}))