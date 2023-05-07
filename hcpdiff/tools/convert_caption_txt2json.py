import argparse
import json
import os

from hcpdiff.utils.img_size_tool import types_support

parser = argparse.ArgumentParser(description='Stable Diffusion Training')
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--with_imgs', action="store_true")
args = parser.parse_args()


def get_txt_caption(path):
    with open(path, encoding='utf8') as f:
        return f.read().strip()


captions = {}
for file in os.listdir(args.data_root):
    file_name, file_ext = file.rsplit('.', 1)
    if args.with_imgs:
        if file_ext in types_support:
            captions[file] = get_txt_caption(os.path.join(args.data_root, f'{file_name}.txt'))
    else:
        if file_ext == 'txt':
            captions[file] = get_txt_caption(os.path.join(args.data_root, f'{file_name}.txt'))

with open(os.path.join(args.data_root, f'image_captions.json'), "w", encoding='utf8') as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)
