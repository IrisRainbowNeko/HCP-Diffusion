
class NoiseBase:
    def __init__(self, base_scheduler):
        self.base_scheduler = base_scheduler

    def __getattr__(self, item):
        try:
            return super(NoiseBase, self).__getattr__(item)
        except:
            return getattr(self.base_scheduler, item)

    def __setattr__(self, key, value):
        if  hasattr(super(), 'base_scheduler') and hasattr(self.base_scheduler, key):
            setattr(self.base_scheduler, key, value)
        else:
            super(NoiseBase, self).__setattr__(key, value)
