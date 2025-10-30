import numpy as np
from scipy.integrate import solve_ivp


class RollingSolver:
    def __init__(self, g: float = 9.81):
        self.g = g

    def solve(self, model, initial_state, t_span, t_eval):
        solution = solve_ivp(model, t_span, initial_state, t_eval=t_eval, method='RK45')
        return solution

