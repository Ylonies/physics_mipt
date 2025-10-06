import numpy as np
from scipy.integrate import solve_ivp

class TrajectorySolver:
    def __init__(self, g=9.81):
        self.g = g
    
    def solve(self, model, params):
        v0 = params['v0']
        alpha = params['alpha']
        m = params['m']

        v0_x = v0 * np.cos(alpha)
        v0_y = v0 * np.sin(alpha)
        initial_state = [0, 0, v0_x, v0_y]

        if alpha == 0:
            initial_state[1] = 0.00000000001  # начальная высота близка к 0 но не равна, чтобы вычисление заранее не завершалось из за нахождения в нуле
            t_base = np.sqrt(2 * 0.00000000001 / self.g)
        else:
            t_base = 2 * v0 * np.sin(alpha) / self.g
        resistance_factor = 1 + 2 * (params['gamma'] / m)
        t_max = t_base * resistance_factor
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, 100000)
        
        def hit_ground(t, state):
            return state[1]
        hit_ground.terminal = True
        hit_ground.direction = -1
        
        solution = solve_ivp(model, t_span, initial_state, 
                           t_eval=t_eval, events=hit_ground, 
                           method='RK45')
        
        return solution