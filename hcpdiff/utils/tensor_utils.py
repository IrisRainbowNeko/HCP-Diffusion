import torch

class ListTensor(list):
    def to(self, device=None, dtype=None):
        for i, v in enumerate(self):
            self[i] = v.to(device=device, dtype=dtype)
        return self

    def squeeze(self, dim=0):
        for i, v in enumerate(self):
            self[i] = v.squeeze(dim)
        return self

    @classmethod
    def create(cls, data):
        if isinstance(data[0], torch.Tensor):
            return cls(data)
        else:
            return data