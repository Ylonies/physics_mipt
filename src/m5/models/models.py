import math
import numpy as np

class PhysicsModels:
    def __init__(self, g: float = 9.81):
        self.g = g

    def ellipse_pendulum_model(self, params):
        m = params['m']
        l = params['l']
        I = params['I']
        gamma = params['gamma']

        def model(t, y):
            theta, omega = y
            dtheta_dt = omega
            domega_dt = -(m * self.g * l / I) * math.sin(theta) - gamma/I * omega
            return [dtheta_dt, domega_dt]

        y0 = [params['theta0'], params['omega0']]
        t_span = (0.0, params['t_end'])
        t_eval = np.linspace(*t_span, 2000)
        return model, y0, t_span, t_eval

