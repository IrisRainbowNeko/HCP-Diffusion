import argparse
import json
import os.path
from typing import Callable

import pyarrow.parquet as pq
import torch
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from tqdm.auto import tqdm

from hcpdiff.data.caption_loader import auto_caption_loader

class DatasetCreator:
    def __init__(self, pretrained_model, out_dir: str, img_w: int=512, img_h: int=512):
        scheduler = DPMSolverMultistepScheduler(
            beta_start = 0.00085,
            beta_end = 0.012,
            beta_schedule = 'scaled_linear',
            algorithm_type = 'dpmsolver++',
            use_karras_sigmas = True,
        )

        self.pipeline = DiffusionPipeline.from_pretrained(pretrained_model, scheduler=scheduler, torch_dtype=torch.float16)
        self.pipeline.requires_safety_checker = False
        self.pipeline.safety_checker = None
        self.pipeline.to("cuda")
        self.pipeline.unet.to(memory_format=torch.channels_last)
        #self.pipeline.enable_xformers_memory_efficient_attention()

        self.out_dir = out_dir
        self.img_w = img_w
        self.img_h = img_h

    def create_from_prompt_dataset(self, prompt_file: str, negative_prompt: str, bs: int, num: int=None, repeat:int=1, save_fmt:str='txt',
                                   callback: Callable[[int, int], bool] = None):
        os.makedirs(self.out_dir, exist_ok=True)
        data = auto_caption_loader(prompt_file).load()
        data = list(data.items())
        data = self.split_batch(data, bs)  # [[(k,v),...],...]

        if num is None:
            num = len(data)
        total = num*bs
        count = 0
        captions = {}
        with torch.inference_mode():
            for i in tqdm(range(num)):
                for r in range(repeat):
                    name_batch, p_batch = list(zip(*data[i%len(data)]))
                    imgs = self.pipeline(list(p_batch), negative_prompt=[negative_prompt]*len(p_batch), num_inference_steps=25,
                                         width=self.img_w, height=self.img_h).images
                    for name, prompt, img in zip(name_batch, p_batch, imgs):
                        img.save(os.path.join(self.out_dir, f'{count}_{name}.png'), format='PNG')
                        captions[f'{count}_{name}'] = prompt
                        count += 1
                    if callback:
                        if not callback(count, total):
                            break

        if save_fmt=='txt':
            for k, v in captions.items():
                with open(os.path.join(self.out_dir, f'{k}.txt'), "w") as f:
                    f.write(v)
        elif save_fmt=='json':
            with open(os.path.join(self.out_dir, f'image_captions.json'), "w") as f:
                json.dump(captions, f)
        else:
            raise ValueError(f"Invalid save_fmt: {save_fmt}")

    @staticmethod
    def split_batch(data, bs):
        return [data[i:i+bs] for i in range(0, len(data), bs)]

# python dataset_generator.py --prompt_file 标注文件或文件夹路径 --model 模型名称 --out_dir 输出文件夹路径 --repeat 每个prompt生成几个图 --bs batch_size --img_w 图片宽度 --img_h 图片高度
# python dataset_generator.py --prompt_file 标注文件或文件夹路径 --model 模型名称 --out_dir 输出文件夹路径 --repeat 1 --bs 4 --img_w 640 --img_h 640
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='Diffusion Dataset Generator')
    parser.add_argument('--prompt_file', type=str, default='')
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--out_dir', type=str, default=r'./prompt_ds')
    parser.add_argument('--negative_prompt', type=str,
                        default='lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry')
    parser.add_argument('--num', type=int, default=200)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--save_fmt', type=str, default='txt')
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--img_h', type=int, default=512)
    args = parser.parse_args()

    ds_creator = DatasetCreator(args.model, args.out_dir, args.img_w, args.img_h)
    ds_creator.create_from_prompt_dataset(args.prompt_file, args.negative_prompt, args.bs, args.num, repeat=args.repeat, save_fmt=args.save_fmt)
