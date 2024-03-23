from typing import List, Dict, Callable
from tqdm.auto import tqdm

class from_memory:
    def __init__(self, memory, mem_name):
        self.mem_name = mem_name
        self.memory = memory

    def __call__(self):
        memory = self.memory  # use in eval
        return eval(f'memory.{self.mem_name}')

def from_memory_context(fun):
    def f(*args, **kwargs):
        filter_kwargs = {k:(v() if isinstance(v, from_memory) else v) for k, v in kwargs.items()}
        return fun(*args, **filter_kwargs)

    return f

def feedback_input(fun, exclude_keys=('memory',)):
    def f(*args, **states):
        output = fun(*args, **states)
        for key in exclude_keys:
            if key in states:
                del states[key]
        if output is not None:
            for key in exclude_keys:
                if key in output:
                    del output[key]

            if '_ex_input' in output:
                if '_ex_input' not in states:
                    states['_ex_input'] = output['_ex_input']
                else:
                    states['_ex_input'].update(output['_ex_input'])
                del output['_ex_input']

            states.update(output)
        return states

    return f

class BasicAction:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class ExecAction(BasicAction):
    def __init__(self, prog: str):
        self.prog = prog

    def forward(self, memory, **states):
        exec(self.prog)
        return states

class LambdaAction(BasicAction):
    def __init__(self, func: Callable):
        self.func = func

    def forward(self, memory, **states):
        states = self.func(memory=memory, **states)
        return states

class LoopAction(BasicAction):
    def __init__(self, loop_value: Dict[str, str], actions: List[BasicAction]):
        self.loop_value = loop_value
        self.actions = actions

    def forward(self, memory, **states):
        loop_data = [states.pop(k) for k in self.loop_value.keys()]
        pbar = tqdm(zip(*loop_data), total=len(loop_data[0]))
        N_steps = len(self.actions)
        for data in pbar:
            feed_data = {k:v for k, v in zip(self.loop_value.values(), data)}
            states.update(feed_data)
            for step, act in enumerate(self.actions):
                pbar.set_description(f'[{step+1}/{N_steps}] action: {type(act).__name__}')
                states = act(memory=memory, **states)
        return states
