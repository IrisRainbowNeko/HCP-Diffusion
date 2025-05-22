from rainbowneko.ckpt_manager import NekoSaver, CkptFormat, LocalCkptSource, PKLFormat, LAYERS_ALL, LAYERS_TRAINABLE
from torch import Tensor
from typing import Dict, Any

class EmbSaver(NekoSaver):
    def __init__(self, format: CkptFormat=None, source: LocalCkptSource=None, layers='all', key_map=None, prefix=None):
        if format is None:
            format = PKLFormat()
        if source is None:
            source = LocalCkptSource()
        key_map = key_map or ('name -> name', 'embs -> embs', 'name_template -> name_template')
        super().__init__(format, source, layers=layers, key_map=key_map)
        self.prefix = prefix

    def _save_to(self, name, embs: Dict[str, Tensor], name_template=None):
        for pt_name, pt in embs.items():
            if self.layers == LAYERS_ALL:
                pass
            elif self.layers == LAYERS_TRAINABLE:
                if not pt.requires_grad:
                    continue
            elif pt_name not in self.layers:
                continue

            self.save((pt_name, pt), pt_name, prefix=self.prefix)
            if name_template is not None:
                pt_name = name_template.format(pt_name)
                self.save((pt_name, pt), pt_name, prefix=self.prefix)
