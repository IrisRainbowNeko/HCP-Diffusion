from .base import Sampler

class VPSampler(Sampler):
    # closed-form: \alpha(t)^2 + \sigma(t)^2 = 1
    def velocity_to_eps(self, v_pred, x_t, t):
        alpha = self.sigma_scheduler.alpha(t)
        sigma = self.sigma_scheduler.sigma(t)
        return alpha*v_pred+sigma*x_t

    def eps_to_velocity(self, eps, x_t, t, x_0=None):
        alpha = self.sigma_scheduler.alpha(t)
        sigma = self.sigma_scheduler.sigma(t)
        if x_0 is None:
            x_0 = self.eps_to_x0(eps, x_t, t)
        return alpha*eps-sigma*x_0

    def velocity_to_x0(self, v_pred, x_t, t):
        alpha = self.sigma_scheduler.alpha(t)
        sigma = self.sigma_scheduler.sigma(t)
        return alpha*x_t-sigma*v_pred

    def x0_to_velocity(self, x_0, x_t, t, eps=None):
        alpha = self.sigma_scheduler.alpha(t)
        sigma = self.sigma_scheduler.sigma(t)
        if eps is None:
            eps = self.x0_to_eps(x_0, x_t, t)
        return alpha*eps-sigma*x_0