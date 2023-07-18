import argparse
import json
import os.path
from typing import Callable

import pyarrow.parquet as pq
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm

class DatasetCreator:
    def __init__(self, pretrained_model, out_dir: str, img_w: int=512, img_h: int=512):
        self.pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16)
        self.pipeline.requires_safety_checker = False
        self.pipeline.safety_checker = None
        self.pipeline.to("cuda")
        self.pipeline.unet.to(memory_format=torch.channels_last)
        self.pipeline.enable_xformers_memory_efficient_attention()

        self.out_dir = out_dir
        self.img_w = img_w
        self.img_h = img_h

    def create_from_prompt_dataset(self, prompt_file: str, negative_prompt: str, bs: int, num: int,
                                   callback: Callable[[int, int], bool] = None):
        os.makedirs(self.out_dir, exist_ok=True)
        data = pq.read_table(prompt_file).to_batches(bs)

        count = 0
        total = num*bs
        captions = {}
        with torch.inference_mode():
            for i in tqdm(range(num)):
                p_batch = data[i][0].to_pylist()
                imgs = self.pipeline(p_batch, negative_prompt=[negative_prompt]*len(p_batch), width=self.img_w, height=self.img_h).images
                for prompt, img in zip(p_batch, imgs):
                    img.save(os.path.join(self.out_dir, f'{count}.png'), format='PNG')
                    captions[f'{count}.png'] = prompt
                    count += 1
                if callback:
                    if not callback(count, total):
                        break

        with open(os.path.join(self.out_dir, f'image_captions.json'), "w") as f:
            json.dump(captions, f)
            import huggingface_hub.utils._validators

    @staticmethod
    def split_batch(data, bs):
        return [data[i:i+bs] for i in range(0, len(data), bs)]

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--prompt_file', type=str, default='')
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--out_dir', type=str, default=r'./prompt_ds')
    parser.add_argument('--negative_prompt', type=str,
                        default='lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry')
    parser.add_argument('--num', type=int, default=200)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--img_h', type=int, default=512)
    args = parser.parse_args()

    ds_creator = DatasetCreator(args.model, args.out_dir, args.img_w, args.img_h)
    ds_creator.create_from_prompt_dataset(args.prompt_file, args.negative_prompt, args.bs, args.num)
