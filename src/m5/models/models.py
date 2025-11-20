import math
import numpy as np

class PhysicsModels:
    def __init__(self, g: float = 9.81):
        self.g = g

    def solve_symplectic(self, params):
        m = params['m']
        l = params['l']
        I = params['I']
        gamma = params['gamma']   
        t_end = params['t_end']

        dt = t_end / 2000
        steps = int(t_end / dt)

        theta = np.zeros(steps)
        omega = np.zeros(steps)
        time = np.zeros(steps)

        theta[0] = params['theta0']
        omega[0] = params['omega0']

        def accel(th, om):
            return -(m * self.g * l / I) * np.sin(th) - gamma / I * om

        for i in range(steps - 1):
            omega_half = omega[i] + 0.5 * dt * accel(theta[i], omega[i])
            theta[i + 1] = theta[i] + dt * omega_half
            omega[i + 1] = omega_half + 0.5 * dt * accel(theta[i + 1], omega_half)
            time[i + 1] = time[i] + dt

        return time, theta, omega
