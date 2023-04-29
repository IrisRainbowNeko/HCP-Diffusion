import os.path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import pyarrow.parquet as pq
import argparse
import json
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='Stable Diffusion Training')
parser.add_argument('--prompt_file', type=str, default='')
parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5')
parser.add_argument('--out_dir', type=str, default=r'./prompt_ds')
parser.add_argument('--negative_prompt', type=str, default='')
parser.add_argument('--num', type=int, default=200)
parser.add_argument('--bs', type=int, default=4)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True


def split_batch(data, bs):
    return [data[i:i + bs] for i in range(0, len(data), bs)]


data = pq.read_table(args.prompt_file).to_batches(args.bs)
pipeline = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
pipeline.requires_safety_checker = False
pipeline.safety_checker = None
pipeline.to("cuda")
pipeline.unet.to(memory_format=torch.channels_last)
pipeline.enable_xformers_memory_efficient_attention()

count = 0
captions = {}
with torch.inference_mode():
    for i in tqdm(range(args.num)):
        p_batch = data[i][0].to_pylist()
        imgs = pipeline(p_batch, negative_prompt=[args.negative_prompt] * len(p_batch)).images
        for prompt, img in zip(p_batch, imgs):
            img.resize((512, 512), Image.BILINEAR).save(os.path.join(args.out_dir, f'{count}.png'), format='PNG')
            captions[f'{count}.png'] = prompt
            count += 1

with open(os.path.join(args.out_dir, f'image_captions.json'), "w") as f:
    json.dump(captions, f)
