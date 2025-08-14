from .diffusion import InputFeederAction, MakeLatentAction, SD15DenoiseAction, SDXLDenoiseAction, PixartDenoiseAction, FluxDenoiseAction, SampleAction, DiffusionStepAction, \
    X0PredAction, SeedAction, MakeTimestepsAction, PrepareDiffusionAction, time_iter, DiffusionActions
from .text import TextEncodeAction, TextHookAction, AttnMultTextEncodeAction
from .vae import EncodeAction, DecodeAction
from .io import BuildModelsAction, SaveImageAction, LoadImageAction
from .utils import LatentResizeAction, ImageResizeAction, FeedtoCNetAction
from .model import VaeOptimizeAction, BuildOffloadAction, XformersEnableAction
#from .flow import FilePromptAction

try:
    from .fast import SFastCompileAction
except:
    print('stable fast not installed.')

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("hcp.from_memory", lambda mem_name:OmegaConf.create({
    '_target_':'hcpdiff.workflow.from_memory',
    'mem_name':mem_name,
}))
