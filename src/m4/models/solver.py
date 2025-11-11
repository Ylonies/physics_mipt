import numpy as np
from scipy.integrate import solve_ivp


class RollingSolver:
    def __init__(self, g: float = 9.81):
        self.g = g

    def solve(self, model, initial_state, t_span, t_eval):
        t_start, t_end = t_span
        duration = t_end - t_start

        if duration <= 10:
            rtol, atol, n_points = 1e-8, 1e-10, 3000
        elif duration <= 100:
            rtol, atol, n_points = 1e-6, 1e-8, 2000
        elif duration <= 1000:
            rtol, atol, n_points = 1e-5, 1e-7, 1000
        else:
            rtol, atol, n_points = 1e-4, 1e-6, 500

        t_eval = np.linspace(t_start, t_end, n_points)

        solution = solve_ivp(
            model, t_span, initial_state,
            t_eval=t_eval, method='RK45',
            rtol=rtol, atol=atol
        )
        return solution
