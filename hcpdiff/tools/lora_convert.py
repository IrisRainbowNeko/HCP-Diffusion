import argparse
import os.path
from typing import List
import math

from hcpdiff.ckpt_manager import auto_manager

class LoraConverter:
    com_name_unet = ['down_blocks', 'up_blocks', 'mid_block', 'transformer_blocks', 'to_q', 'to_k', 'to_v', 'to_out', 'proj_in', 'proj_out']
    com_name_TE = ['self_attn', 'q_proj', 'v_proj', 'k_proj', 'out_proj', 'text_model']
    prefix_unet = 'lora_unet_'
    prefix_TE = 'lora_te_'

    def __init__(self):
        self.com_name_unet_tmp = [x.replace('_', '%') for x in self.com_name_unet]
        self.com_name_TE_tmp = [x.replace('_', '%') for x in self.com_name_TE]

    def convert_from_webui(self, state, auto_scale_alpha=False):
        sd_unet = self.convert_from_webui_(state, prefix=self.prefix_unet, com_name=self.com_name_unet, com_name_tmp=self.com_name_unet_tmp)
        sd_TE = self.convert_from_webui_(state, prefix=self.prefix_TE, com_name=self.com_name_TE, com_name_tmp=self.com_name_TE_tmp)
        if auto_scale_alpha:
            sd_unet = self.alpha_scale_from_webui(sd_unet)
            sd_TE = self.alpha_scale_from_webui(sd_TE)
        return {'lora': sd_TE},  {'lora': sd_unet}

    def convert_to_webui(self, sd_unet, sd_TE, auto_scale_alpha=False):
        sd_unet = self.convert_to_webui_(sd_unet, prefix=self.prefix_unet)
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
                sd_covert[f'{model_k}.___.layer.{lora_k}'] = v
        return sd_covert

    def convert_to_webui_(self, state, prefix):
        sd_covert = {}
        for k, v in state.items():
            model_k, lora_k = k.split('.___.' if ('alpha' in k or 'scale' in k) else '.___.layer.', 1)
            sd_covert[f"{prefix}{model_k.replace('.', '_')}.{lora_k}"] = v
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
            if 'lora_up' in k:
                state[k] = v*math.sqrt(v.shape[1])
            elif 'lora_down' in k:
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
    args = parser.parse_args()

    converter = LoraConverter()
    lora_name = os.path.basename(args.lora_path)

    # load lora model
    print('convert lora model')
    ckpt_manager = auto_manager(args.lora_path)()

    if args.from_webui:
        state = ckpt_manager.load_ckpt(args.lora_path)
        # convert the weight name
        sd_TE, sd_unet = converter.convert_from_webui(state, auto_scale_alpha=args.auto_scale_alpha)
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
        sd_TE = ckpt_manager.load_ckpt(args.lora_path_TE)
        state = converter.convert_to_webui(sd_unet['lora'], sd_TE['lora'], auto_scale_alpha=args.auto_scale_alpha)
        ckpt_manager._save_ckpt(state, save_path=args.dump_path)
        print('save lora to:', args.dump_path)
