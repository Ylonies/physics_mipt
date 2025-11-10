import numpy as np

class ResultAnalyzer:
    @staticmethod
    def analyze(solution, params):
        theta = solution.y[0]
        omega = solution.y[1]
        m = params['m']
        l = params['l']
        I = params['I']
        g = 9.81

        kinetic = 0.5 * I * omega**2
        potential = m * g * l * (1 - np.cos(theta))
        energy_total = kinetic + potential

        return {
            'time': solution.t,
            'theta': theta,
            'omega': omega,
            'energy_total': energy_total
        }