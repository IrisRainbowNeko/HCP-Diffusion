from .base import BasicAction, from_memory_context, feedback_input, ContainerAction
from typing import List, Dict
from tqdm import tqdm
import math

class FilePromptAction(BasicAction):
    def __init__(self, actions: List[BasicAction], prompt: str, negative_prompt: str, bs: int = 4, num: int = None):
        super().__init__()
        if prompt.endswith('.txt'):
            with open(prompt, 'r') as f:
                prompt = f.read().split('\n')
            if num:
                prompt = prompt[:num]
        else:
            prompt = [prompt]
            if num:
                prompt = prompt*num

        if negative_prompt.endswith('.txt'):
            with open(negative_prompt, 'r') as f:
                negative_prompt = f.read().split('\n')
        else:
            negative_prompt = [negative_prompt]*len(prompt)

        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.bs = bs
        self.actions = actions


    @feedback_input
    def forward(self, memory, **states):
        states.update({'prompt_all':self.prompt, 'negative_prompt_all':self.negative_prompt})

        pbar = tqdm(range(math.ceil(len(self.prompt)/self.bs)))
        N_steps = len(self.actions)
        for gen_step in pbar:
            feed_data = {'gen_step': gen_step}
            states.update(feed_data)
            for step, act in enumerate(self.actions):
                pbar.set_description(f'[{step+1}/{N_steps}] action: {type(act).__name__}')
                states = act(memory=memory, **states)
        return states