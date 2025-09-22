from .folder_class import T2IFolderClassSource
from .text import TextSource
from .text2img import Text2ImageSource, Text2ImageLossMapSource
from .text2img_cond import Text2ImageCondSource

try:
    from .text2img import WebDSText2ImageSource
except ImportError:
    pass
