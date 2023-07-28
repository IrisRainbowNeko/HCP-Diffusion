#TODO 补齐数据说明

class Sd2diffusers_convert_args:
    clip_stats_path:str         #不认得
    controlnet:str              #控制网文件路径
    device:str                  #设备（？）
    image_size:str              #图片尺寸(用于区分模型版本,512对应V1.X,768对应2.X,1024对应XL)
    num_in_channels:str         #频道数（？）
    pipeline_type:str           #不认得
    prediction_type:str         #不认得
    pipeline_type:str           #不认得
    scheduler_type:str          #不认得
    stable_unclip:str           #不认得
    stable_unclip_prior:str     #不认得
    extract_ema:bool            #导出ema模型
    from_safetensors:bool       #源模型是safetensors
    to_safetensors:bool         #导出safetensors
    half:bool                   #导出半精度（fp16）
    upcast_attention:bool       #不认得

    def __init__(self,args:dict|None=None):
        if args:
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
        else:
            self.clip_stats_path=None
            self.controlnet=None
            self.device=None
            self.image_size=None
            self.num_in_channels=None
            self.pipeline_type=None
            self.prediction_type=None
            self.pipeline_type=None
            self.scheduler_type=None
            self.stable_unclip=None
            self.stable_unclip_prior=None
            self.extract_ema=False
            self.from_safetensors=False
            self.to_safetensors=False
            self.half=False
            self.upcast_attention=False