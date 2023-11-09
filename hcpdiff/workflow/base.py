from typing import List, Dict
from tqdm.auto import tqdm

class from_memory:
    #TODO: add memory for all from_memory in cfg
    def __init__(self, memory, mem_name):
        self.mem_name = mem_name
        self.memory = memory

    def __call__(self):
        memory = self.memory # use in eval
        return eval(f'memory.{self.mem_name}')

def from_memory_context(fun):
    def f(*args, **kwargs):
        filter_kwargs = {k: (v() if isinstance(v, from_memory) else v) for k,v in kwargs.items()}
        return fun(*args, **filter_kwargs)
    return f

class BasicAction:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class MemoryMixin:
    pass

class ExecAction(BasicAction, MemoryMixin):
    def __init__(self, prog:str):
        self.prog = prog

    def forward(self, memory, **states):
        exec(self.prog)
        return states

class LoopAction(BasicAction, MemoryMixin):
    def __init__(self, loop_value:Dict[str, str], actions:List[BasicAction]):
        self.loop_value = loop_value
        self.actions = actions

    def forward(self, memory, **states):
        loop_data = [states.pop(k) for k in self.loop_value.keys()]
        pbar = tqdm(zip(*loop_data), total=len(loop_data[0]))
        N_steps = len(self.actions)
        for data in pbar:
            feed_data = {k:v for k,v in zip(self.loop_value.values(), data)}
            states.update(feed_data)
            for step, act in enumerate(self.actions):
                pbar.set_description(f'[{step+1}/{N_steps}] action: {type(act).__name__}')
                if isinstance(act, MemoryMixin):
                    states = act(memory=memory, **states)
                else:
                    states = act(**states)
        return states
