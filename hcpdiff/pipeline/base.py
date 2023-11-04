
class from_memory:
    #TODO: add memory for all from_memory in cfg
    def __init__(self, memory, mem_name):
        self.mem_name = mem_name
        self.memory = memory

    def __call__(self):
        memory = self.memory # use in eval
        return eval(self.mem_name)

def from_memory_context(fun):
    def f(*args, **kwargs):
        filter_kwargs = {}
        for k,v in kwargs.items():
            if isinstance(v, from_memory):
                filter_kwargs[k] = v()
            else:
                filter_kwargs[k] = v
        return fun(*args, **filter_kwargs)
    return f

class BasicAction:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class ExecAction(BasicAction):
    def __init__(self, prog):
        self.prog = prog

    def forward(self, memory, **states):
        exec(self.prog)