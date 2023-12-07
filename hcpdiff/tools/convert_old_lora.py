import torch
from typing import Dict, Any

def convert_state(lora_state: Dict[str, Any]):
    new_state = {}

    new_state['layer.W_down'] = lora_state['layer.lora_down.weight']
    new_state['layer.W_up'] = lora_state['layer.lora_up.weight']
    if 'layer.lora_up.bias' in lora_state:
        new_state['layer.bias'] = lora_state['layer.lora_up.bias']
    if 'alpha' in lora_state:
        new_state['alpha'] = lora_state['alpha']
    return new_state