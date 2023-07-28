#TODO 补齐数据说明

class Sd2diffusers_convert_args:
    clip_stats_path:str=None         #不认得
    vae_pt_path:str=None             #VAE文件路径
    controlnet:str=None              #控制网文件路径
    device:str=None                  #设备（？）
    image_size:str=None              #图片尺寸(用于区分模型版本,512对应V1.X,768对应2.X,1024对应XL)
    num_in_channels:str=None         #频道数（？）
    pipeline_type:str=None           #不认得
    prediction_type:str=None         #不认得
    pipeline_type:str=None           #不认得
    scheduler_type:str=None          #不认得
    stable_unclip:str=None           #不认得
    stable_unclip_prior:str=None     #不认得
    extract_ema:bool=False           #导出ema模型
    from_safetensors:bool=False      #源模型是safetensors
    to_safetensors:bool=False        #导出safetensors
    half:bool=False                  #导出半精度（fp16）
    upcast_attention:bool=False      #不认得

    def __init__(self):
        pass
    
    def __init__(self,args):
        self.clip_stats_path=args.clip_stats_path
        self.vae_pt_path=args.vae_pt_path
        self.controlnet=args.controlnet
        self.device=args.device
        self.image_size=args.image_size
        self.num_in_channels=args.num_in_channels
        self.pipeline_type=args.pipeline_type
        self.prediction_type=args.prediction_type
        self.pipeline_type=args.pipeline_type
        self.scheduler_type=args.scheduler_type
        self.stable_unclip=args.stable_unclip
        self.stable_unclip_prior=args.stable_unclip_prior
        self.extract_ema=args.extract_ema
        self.from_safetensors=args.from_safetensors
        self.to_safetensors=args.to_safetensors
        self.half=args.half
        self.upcast_attention=args.upcast_attention