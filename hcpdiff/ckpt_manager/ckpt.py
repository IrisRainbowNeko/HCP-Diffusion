from rainbowneko.ckpt_manager import NekoSaver, CkptFormat, LocalCkptSource, PKLFormat
from torch import nn
from typing import Dict, Any

class EmbSaver(NekoSaver):
    def __init__(self, format: CkptFormat, source: LocalCkptSource, target_key='embs', prefix=None):
        super().__init__(format, source)
        self.target_key = target_key
        self.prefix = prefix

    def save_to(self, name, model: nn.Module, plugin_groups: Dict[str, Any], model_ema=None, exclude_key=None,
                name_template=None):
        train_pts = plugin_groups[self.target_key]
        for pt_name, pt in train_pts.items():
            self.save(pt_name, (pt_name, pt), prefix=self.prefix)
            if name_template is not None:
                pt_name = name_template.format(pt_name)
                self.save(pt_name, (pt_name, pt), prefix=self.prefix)

def easy_emb_saver():
    return EmbSaver(
        format=PKLFormat(),
        source=LocalCkptSource(),
    )
