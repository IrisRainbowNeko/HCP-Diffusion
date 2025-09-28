from .base import ODESolver

class EulerSolver(ODESolver):
    def step(self, x_t, v_t, dt, **kwargs):
        return x_t - v_t*dt