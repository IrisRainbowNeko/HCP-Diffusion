import argparse
import os.path
from typing import List
import math

from hcpdiff.ckpt_manager import auto_manager
from hcpdiff.deprecated import convert_to_webui_maybe_old, convert_to_webui_xl_maybe_old

class LoraConverter:
    com_name_unet = ['down_blocks', 'up_blocks', 'mid_block', 'transformer_blocks', 'to_q', 'to_k', 'to_v', 'to_out', 'proj_in', 'proj_out', 'input_blocks', 'middle_block', 'output_blocks']
    com_name_TE = ['self_attn', 'q_proj', 'v_proj', 'k_proj', 'out_proj', 'text_model']
    prefix_unet = 'lora_unet_'
    prefix_TE = 'lora_te_'
    prefix_TE_xl_clip_B = 'lora_te1_'
    prefix_TE_xl_clip_bigG = 'lora_te2_'

    lora_w_map = {'lora_down.weight': 'W_down', 'lora_up.weight':'W_up'}

    def __init__(self):
        self.com_name_unet_tmp = [x.replace('_', '%') for x in self.com_name_unet]
        self.com_name_TE_tmp = [x.replace('_', '%') for x in self.com_name_TE]

    def convert_from_webui(self, state, auto_scale_alpha=False, sdxl=False):
        if not sdxl:
            sd_unet = self.convert_from_webui_(state, prefix=self.prefix_unet, com_name=self.com_name_unet, com_name_tmp=self.com_name_unet_tmp)
            sd_TE = self.convert_from_webui_(state, prefix=self.prefix_TE, com_name=self.com_name_TE, com_name_tmp=self.com_name_TE_tmp)
        else:
            sd_unet = self.convert_from_webui_xl_unet_(state, prefix=self.prefix_unet, com_name=self.com_name_unet, com_name_tmp=self.com_name_unet_tmp)
            sd_TE = self.convert_from_webui_xl_te_(state, prefix=self.prefix_TE_xl_clip_B, com_name=self.com_name_TE, com_name_tmp=self.com_name_TE_tmp)
            sd_TE2 = self.convert_from_webui_xl_te_(state, prefix=self.prefix_TE_xl_clip_bigG, com_name=self.com_name_TE, com_name_tmp=self.com_name_TE_tmp)
            sd_TE.update(sd_TE2)

        if auto_scale_alpha:
            sd_unet = self.alpha_scale_from_webui(sd_unet)
            sd_TE = self.alpha_scale_from_webui(sd_TE)
        return {'lora': sd_TE},  {'lora': sd_unet}

    def convert_to_webui(self, sd_unet, sd_TE, auto_scale_alpha=False, sdxl=False):
        sd_unet = self.convert_to_webui_(sd_unet, prefix=self.prefix_unet)
        if sdxl:
            sd_TE = self.convert_to_webui_xl_(sd_TE, prefix=self.prefix_TE)
        else:
            sd_TE = self.convert_to_webui_(sd_TE, prefix=self.prefix_TE)
        sd_unet.update(sd_TE)
        if auto_scale_alpha:
            sd_unet = self.alpha_scale_to_webui(sd_unet)
        return sd_unet

    def convert_from_webui_(self, state, prefix, com_name, com_name_tmp):
        state = {k:v for k, v in state.items() if k.startswith(prefix)}
        prefix_len = len(prefix)
        sd_covert = {}
        for k, v in state.items():
            model_k, lora_k = k[prefix_len:].split('.', 1)
            model_k = self.replace_all(model_k, com_name, com_name_tmp).replace('_', '.').replace('%', '_')
            if lora_k == 'alpha':
                sd_covert[f'{model_k}.___.{lora_k}'] = v
            else:
                sd_covert[f'{model_k}.___.layer.{self.lora_w_map[lora_k]}'] = v
        return sd_covert

    @convert_to_webui_maybe_old
    def convert_to_webui_(self, state, prefix):
        sd_covert = {}
        for k, v in state.items():
            if k.endswith('W_down'):
                model_k, _ = k.split('.___.', 1)
                lora_k = 'lora_down.weight'
            elif k.endswith('W_up'):
                model_k, _ = k.split('.___.', 1)
                lora_k = 'lora_up.weight'
            else:
                model_k, lora_k = k.split('.___.', 1)

            sd_covert[f"{prefix}{model_k.replace('.', '_')}.{lora_k}"] = v
        return sd_covert

    @convert_to_webui_xl_maybe_old
    def convert_to_webui_xl_(self, state, prefix):
        sd_convert = {}
        for k, v in state.items():
            if k.endswith('W_down'):
                model_k, _ = k.split('.___.', 1)
                lora_k = 'lora_down.weight'
            elif k.endswith('W_up'):
                model_k, _ = k.split('.___.', 1)
                lora_k = 'lora_up.weight'
            else:
                model_k, lora_k = k.split('.___.', 1)

            new_k = f"{prefix}{model_k.replace('.', '_')}.{lora_k}"
            if 'clip' in new_k:
                new_k = new_k.replace('_clip_B', '1') if 'clip_B' in new_k else new_k.replace('_clip_bigG', '2')
            sd_convert[new_k] = v
        return sd_convert
    
    def convert_from_webui_xl_te_(self, state, prefix, com_name, com_name_tmp):
        state = {k:v for k, v in state.items() if k.startswith(prefix)}
        sd_covert = {}
        prefix_len = len(prefix)

        for k, v in state.items():
            model_k, lora_k = k[prefix_len:].split('.', 1)
            model_k = self.replace_all(model_k, com_name, com_name_tmp).replace('_', '.').replace('%', '_')
            if prefix == 'lora_te1_':
                model_k = f'clip_B.{model_k}'
            else:
                model_k = f'clip_bigG.{model_k}'

            if lora_k == 'alpha':
                sd_covert[f'{model_k}.___.{lora_k}'] = v
            else:
                sd_covert[f'{model_k}.___.layer.{self.lora_w_map[lora_k]}'] = v
        return sd_covert

    def convert_from_webui_xl_unet_(self, state, prefix, com_name, com_name_tmp):
        # Down: 
        # 4 -> 1, 0  4 = 1 + 3 * 1 + 0
        # 5 -> 1, 1  5 = 1 + 3 * 1 + 1
        # 7 -> 2, 0  7 = 1 + 3 * 2 + 0
        # 8 -> 2, 1  8 = 1 + 3 * 2 + 1

        # Up
        # 0 -> 0, 0  0 = 0 * 3 + 0
        # 1 -> 0, 1  1 = 0 * 3 + 1
        # 2 -> 0, 2  2 = 0 * 3 + 2
        # 3 -> 1, 0  3 = 1 * 3 + 0
        # 4 -> 1, 1  4 = 1 * 3 + 1
        # 5 -> 1, 2  5 = 1 * 3 + 2

        down = {
            '4': [1, 0],
            '5': [1, 1],
            '7': [2, 0],
            '8': [2, 1],
        }
        up = {
            '0': [0, 0],
            '1': [0, 1],
            '2': [0, 2],
            '3': [1, 0],
            '4': [1, 1],
            '5': [1, 2],
        }

        import re

        m = []
        def match(key, regex_text):
            regex = re.compile(regex_text)
            r = re.match(regex, key)
            if not r:
                return False

            m.clear()
            m.extend(r.groups())
            return True
        

        state = {k:v for k, v in state.items() if k.startswith(prefix)}
        sd_covert = {}
        prefix_len = len(prefix)
        for k, v in state.items():
            model_k, lora_k = k[prefix_len:].split('.', 1)

            model_k = self.replace_all(model_k, com_name, com_name_tmp).replace('_', '.').replace('%', '_')

            if match(model_k, r'input_blocks.(\d+).1.(.+)'):
                new_k = f'down_blocks.{down[m[0]][0]}.attentions.{down[m[0]][1]}.{m[1]}'
            elif match(model_k, r'middle_block.1.(.+)'):
                new_k = f'mid_block.attentions.0.{m[0]}'
                pass
            elif match(model_k, r'output_blocks.(\d+).(\d+).(.+)'):
                new_k = f'up_blocks.{up[m[0]][0]}.attentions.{up[m[0]][1]}.{m[2]}'
            else:
                raise NotImplementedError

            if lora_k == 'alpha':
                sd_covert[f'{new_k}.___.{lora_k}'] = v
            else:
                sd_covert[f'{new_k}.___.layer.{self.lora_w_map[lora_k]}'] = v

        return sd_covert


    @staticmethod
    def replace_all(data: str, srcs: List[str], dsts: List[str]):
        for src, dst in zip(srcs, dsts):
            data = data.replace(src, dst)
        return data

    @staticmethod
    def alpha_scale_from_webui(state):
        # Apply to "lora_down" and "lora_up" respectively to prevent overflow
        for k, v in state.items():
            if 'W_up' in k:
                state[k] = v*math.sqrt(v.shape[1])
            elif 'W_down' in k:
                state[k] = v*math.sqrt(v.shape[0])
        return state

    @staticmethod
    def alpha_scale_to_webui(state):
        for k, v in state.items():
            if 'lora_up' in k:
                state[k] = v*math.sqrt(v.shape[1])
            elif 'lora_down' in k:
                state[k] = v*math.sqrt(v.shape[0])
        return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default=None, type=str, required=True, help="Path to the lora to convert.")
    parser.add_argument("--lora_path_TE", default=None, type=str, help="Path to the hcp text encoder lora to convert.")
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to save the converted state dict.")
    parser.add_argument("--from_webui", default=None, action="store_true")
    parser.add_argument("--to_webui", default=None, action="store_true")
    parser.add_argument("--auto_scale_alpha", default=None, action="store_true")
    parser.add_argument("--sdxl", default=None, action="store_true")
    args = parser.parse_args()
    
    converter = LoraConverter()
    lora_name = os.path.basename(args.lora_path)

    # load lora model
    print('convert lora model')
    ckpt_manager = auto_manager(args.lora_path)

    if args.from_webui:
        state = ckpt_manager.load_ckpt(args.lora_path)
        # convert the weight name
        sd_TE, sd_unet = converter.convert_from_webui(state, auto_scale_alpha=args.auto_scale_alpha, sdxl=args.sdxl)
        # wegiht save
        os.makedirs(args.dump_path, exist_ok=True)
        TE_path = os.path.join(args.dump_path, 'TE-'+lora_name)
        unet_path = os.path.join(args.dump_path, 'unet-'+lora_name)
        ckpt_manager._save_ckpt(sd_TE, save_path=TE_path)
        ckpt_manager._save_ckpt(sd_unet, save_path=unet_path)
        print('save text encoder lora to:', TE_path)
        print('save unet lora to:', unet_path)
    elif args.to_webui:
        sd_unet = ckpt_manager.load_ckpt(args.lora_path)
        sd_TE = ckpt_manager.load_ckpt(args.lora_path_TE) if args.lora_path_TE else {'lora':{}}
        state = converter.convert_to_webui(sd_unet['lora'], sd_TE['lora'], auto_scale_alpha=args.auto_scale_alpha, sdxl=args.sdxl)
        ckpt_manager._save_ckpt(state, save_path=args.dump_path)
        print('save lora to:', args.dump_path)
