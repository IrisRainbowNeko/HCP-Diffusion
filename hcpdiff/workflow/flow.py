from rainbowneko.infer import BasicAction
from typing import List, Dict
from tqdm import tqdm
import math

class FilePromptAction(BasicAction):
    def __init__(self, actions: List[BasicAction], prompt: str, negative_prompt: str, bs: int = 4, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        if prompt.endswith('.txt'):
            with open(prompt, 'r') as f:
                prompt = f.read().split('\n')
        else:
            prompt = [prompt]

        if negative_prompt.endswith('.txt'):
            with open(negative_prompt, 'r') as f:
                negative_prompt = f.read().split('\n')
        else:
            negative_prompt = [negative_prompt]*len(prompt)

        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.bs = bs
        self.actions = actions


    def forward(self, **states):
        states.update({'prompt_all':self.prompt, 'negative_prompt_all':self.negative_prompt})
        states_ref = dict(**states)

        pbar = tqdm(range(math.ceil(len(self.prompt)/self.bs)))
        N_steps = len(self.actions)
        for gen_step in pbar:
            states = dict(**states_ref)
            feed_data = {'gen_step': gen_step}
            states.update(feed_data)
            for step, act in enumerate(self.actions):
                pbar.set_description(f'[{step+1}/{N_steps}] action: {type(act).__name__}')
                states = act(**states)
        return states

class FlowPromptAction(BasicAction):
    def __init__(self, actions: List[BasicAction], prompt: str, negative_prompt: str, bs: int = 4, num: int = None, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in, key_map_out)
        prompt = [prompt]*num
        negative_prompt = [negative_prompt]*num

        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.bs = bs
        self.actions = actions


    def forward(self, **states):
        states.update({'prompt_all':self.prompt, 'negative_prompt_all':self.negative_prompt})
        states_ref = dict(**states)

        pbar = tqdm(range(math.ceil(len(self.prompt)/self.bs)))
        N_steps = len(self.actions)
        for gen_step in pbar:
            states = dict(**states_ref)
            feed_data = {'gen_step': gen_step}
            states.update(feed_data)
            for step, act in enumerate(self.actions):
                pbar.set_description(f'[{step+1}/{N_steps}] action: {type(act).__name__}')
                states = act(**states)
        return states