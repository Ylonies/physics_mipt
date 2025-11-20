import numpy as np

class ResultAnalyzer:
    @staticmethod
    def analyze(time, theta, omega, params):
        m = params['m']
        l = params['l']
        I = params['I']
        g = 9.81

        K = 0.5 * I * omega**2
        V = m * g * l * (1 - np.cos(theta))
        E = K + V

        return {
            'time': time,
            'theta': theta,
            'omega': omega,
            'K': K,
            'V': V,
            'E': E
        }
