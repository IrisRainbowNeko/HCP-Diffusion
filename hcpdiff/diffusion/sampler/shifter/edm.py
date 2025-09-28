import torch
import numpy as np

class EDMShifter:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def __call__(self, t: torch.Tensor, **kwargs):
        min_inv_rho = self.sigma_min**(1/self.rho)
        max_inv_rho = self.sigma_max**(1/self.rho)
        return torch.lerp(min_inv_rho, max_inv_rho, t)**self.rho

    @property
    def min_dt(self):
        return 1e-8

class EDMRescaleShifter(EDMShifter):
    def __init__(self, ref_scheduler: "SigmaScheduler", rho=7.0):
        ref_t = torch.linspace(0, 1, 1000)
        ref_sigmas = (ref_scheduler.sigma(ref_t)/ref_scheduler.alpha(ref_t))
        super().__init__(ref_sigmas[0], ref_sigmas[1], rho)
        self.ref_scheduler = ref_scheduler
        self.ref_t = ref_t.numpy()
        self.ref_sigmas_log = ref_sigmas.log().numpy()

    def __call__(self, t: torch.Tensor, **kwargs):
        sigma_edm = super().__call__(t)
        t = np.interp(sigma_edm.cpu().clip(min=1e-8).log().numpy(), self.ref_sigmas_log, self.ref_t)
        return torch.tensor(t)

    @property
    def min_dt(self):
        return 1/len(self.ref_t)