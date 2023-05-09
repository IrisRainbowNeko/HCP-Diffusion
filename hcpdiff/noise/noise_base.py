
class NoiseBase:
    def __init__(self, base_scheduler):
        self.base_scheduler = base_scheduler

    def __getattr__(self, item):
        if hasattr(super(), item):
            return super(NoiseBase, self).__getattr__(item)
        else:
            return getattr(self.base_scheduler, item)

    def __setattr__(self, key, value):
        setattr(self.base_scheduler, key, value)