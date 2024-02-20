
def convert_to_webui_maybe_old(new_func):
    def convert_to_webui_(self, state, prefix):
        sd_covert = {}
        for k, v in state.items():
            # new lora format
            if k.endswith('W_down'):
                return new_func(self, state, prefix)

            # old lora format
            model_k, lora_k = k.split('.___.' if ('alpha' in k or 'scale' in k) else '.___.layer.', 1)
            sd_covert[f"{prefix}{model_k.replace('.', '_')}.{lora_k}"] = v
        return sd_covert
    return convert_to_webui_

def convert_to_webui_xl_maybe_old(new_func):
    def convert_to_webui_xl_(self, state, prefix):
        sd_convert = {}
        for k, v in state.items():
            # new lora format
            if k.endswith('W_down'):
                return new_func(self, state, prefix)

            # old lora format
            model_k, lora_k = k.split('.___.' if ('alpha' in k or 'scale' in k) else '.___.layer.', 1)
            new_k = f"{prefix}{model_k.replace('.', '_')}.{lora_k}"
            if 'clip' in new_k:
                new_k = new_k.replace('_clip_B', '1') if 'clip_B' in new_k else new_k.replace('_clip_bigG', '2')
            sd_convert[new_k] = v
        return sd_convert
    return convert_to_webui_xl_