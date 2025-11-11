import numpy as np


class ResultAnalyzer:
    @staticmethod
    def analyze(solution, params, model_name):
        m = params['m']
        R = params['R']
        I = (2.0 / 5.0) * m * R * R
        g = 9.81

        t = solution.t
        y = {}
        for i, key in enumerate(['x', 'y', 'vx', 'vy', 'wx', 'wy']):
            if i < solution.y.shape[0]:
                y[key] = solution.y[i]

        energies = None
        slip_indicator = None

        if 'Наклон' in model_name:
            s = solution.y[0]
            v = solution.y[1]
            omega = solution.y[2]

            height = (0 - s) * np.sin(params['alpha']) 
            potential = m * g * height

            translational = 0.5 * m * v**2
            rotational = 0.5 * I * omega**2
            energies = translational + rotational + potential

            y['s'] = s
            y['v'] = v
            y['omega'] = omega
            y['y'] = (0 - s) * np.sin(params['alpha'])  
        else:
            vx = solution.y[2]
            vy = solution.y[3]
            wx = solution.y[4]
            wy = solution.y[5]
            v2 = vx * vx + vy * vy
            w2 = wx * wx + wy * wy
            energies = 0.5 * m * v2 + 0.5 * I * w2
            slip_indicator = np.linalg.norm(
                np.vstack([vx, vy]).T - R * np.vstack([-wy, wx]).T,
                axis=1,
            ) > 1e-7

        results = {
            'time': t,
            'energy_total': energies,
            'slip_fraction': float(np.mean(slip_indicator)) if slip_indicator is not None else 0.0,
        }
        results.update(y)
        return results

