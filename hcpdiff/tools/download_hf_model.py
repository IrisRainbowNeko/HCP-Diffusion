from diffusers import DiffusionPipeline
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Model')
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--use_safetensors", default=False, action="store_true")
    parser.add_argument("--out_path", type=str, default='ckpts/sd15')
    args = parser.parse_args()

    load_args = dict(torch_dtype = torch.float16 if args.fp16 else torch.float32)
    save_args = dict()

    if args.fp16:
        load_args['variant'] = "fp16"
        save_args['variant'] = "fp16"
    if args.use_safetensors:
        load_args['use_safetensors'] = True
        save_args['safe_serialization'] = True

    pipe = DiffusionPipeline.from_pretrained(args.model, **load_args)
    pipe.save_pretrained(args.out_path, **save_args)