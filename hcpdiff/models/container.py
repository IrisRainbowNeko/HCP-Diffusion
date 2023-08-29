import torch

class ParameterGroup:
    def __init__(self, values):
        self.values = values

    @property
    def requires_grad(self):
        return all(v.requires_grad for v in self.values)

    @requires_grad.setter
    def requires_grad(self, value):
        for v in self.values:
            v.requires_grad = value

    @property
    def data(self):
        return torch.cat([v.data for v in self.values], dim=-1)

    def __getitem__(self, idx):
        return self.values[idx]

    def __repr__(self):
        return 'ParameterGroup:\n' + '\n'.join(repr(v) for v in self.values)