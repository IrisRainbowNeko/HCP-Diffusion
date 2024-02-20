import argparse
import os

import torch
import shutil

from hcpdiff.ckpt_manager import auto_manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--to_webui", default=None, action="store_true")
    parser.add_argument("--from_webui", default=None, action="store_true")
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to save the converted state dict.")
    parser.add_argument("--embedding_path", default=None, type=str)
    parser.add_argument("--sdxl", default=None, type=str)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dump_path), exist_ok=True)

    print(f'convert embedding')
    ckpt_manager = auto_manager(args.embedding_path)
    embedding = ckpt_manager.load_ckpt(args.embedding_path)

    if not args.sdxl:
        shutil.copyfile(args.embedding_path, args.dump_path)
    else:
        if args.to_webui:
            new = embedding['string_to_param']['*']
            new = {'clip_l':new[:, :768], 'clip_g':new[:, 768:]}
            ckpt_manager._save_ckpt(new, save_path=args.dump_path)

        elif args.from_webui:
            new = torch.cat([embedding['clip_l'], embedding['clip_g']], dim=1)
            new = {'string_to_param':{'*':new}}
            ckpt_manager._save_ckpt(new, save_path=args.dump_path)
        else:
            raise ValueError("Either --to_webui or --from_webui should be set.")
    
    print(f'converted embedding saved to {args.dump_path}')