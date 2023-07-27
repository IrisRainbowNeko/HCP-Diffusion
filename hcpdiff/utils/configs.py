from dataclasses import dataclass,field
#TODO 补齐数据说明
@dataclass
class Sd2diffusers_convert_args:
    checkpoint_path:str=field(default=None,metadata="大模型文件路径")
    clip_stats_path:str=field(default=None,metadata="")
    dump_path:str=field(default=None,metadata="输出路径")
    vae_pt_path:str=field(default=None,metadata="VAE文件路径")
    original_config_file:str=field(default=None,metadata="diffusers config文件路径")
    controlnet:str=field(default=None,metadata="控制网文件路径")
    device=field(default=None,metadata="")
    image_size=field(default=None,metadata="图片尺寸(用于区分模型版本,512对应V1.X,768对应2.X,1024对应XL)")
    num_in_channels=field(default=None,metadata="")
    pipeline_type=field(default=None,metadata="")
    prediction_type=field(default=None,metadata="")
    model_type=field(default=None,metadata="")
    scheduler_type=field(default=None,metadata="")
    stable_unclip=field(default=None,metadata="")
    stable_unclip_prior=field(default=None,metadata="")
    extract_ema:bool=field(default=False,metadata="一个布尔参数，用于决定是否导出ema模型")
    from_safetensors:bool=field(default=False,metadata="一个布尔参数，用于决定源模型是否是safetensors")
    to_safetensors:bool=field(default=False,metadata="一个布尔参数，用于决定是否导出safetensors")
    half:bool=field(default=False,metadata="一个布尔参数，用于决定是否导出半精度（fp16）")
    upcast_attention:bool=field(default=False,metadata="")
    #TODO 一个将参数转换成对象的方法
    def convert_parser_args(args):
        pass