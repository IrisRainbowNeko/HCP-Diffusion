from rainbowneko.parser import neko_cfg

@neko_cfg
def make_cfg():
    return dict(
        model_part=None,
        model_plugin=None,
        emb_pt=None
    )
