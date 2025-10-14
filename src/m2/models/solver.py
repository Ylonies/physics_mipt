import numpy as np

class CollisionSolver:
    def __init__(self):
        pass
    
    def solve_elastic_wall(self, params):
        m1 = params['m1']
        v0 = params['v0']
        alpha = params['alpha']
        
        v_after = v0
        alpha_after = alpha
        
        results = {
            'v_after': v_after,
            'alpha_after': alpha_after,
            'vx_after': -v0 * np.cos(alpha),
            'vy_after': v0 * np.sin(alpha)
        }
        
        return results
    
    def solve_elastic_balls(self, params):
        m1 = params['m1']
        m2 = params['m2']
        v0 = params['v0']
        u0 = params['u0']
        
        v1_after = ((m1 - m2) * v0 + 2 * m2 * u0) / (m1 + m2)
        v2_after = ((m2 - m1) * u0 + 2 * m1 * v0) / (m1 + m2)
        
        results = {
            'v1_after': v1_after,
            'v2_after': v2_after,
            'momentum_conserved': abs((m1 * v0 + m2 * u0) - (m1 * v1_after + m2 * v2_after)) < 1e-10,
            'energy_conserved': abs((0.5 * m1 * v0**2 + 0.5 * m2 * u0**2) - 
                                  (0.5 * m1 * v1_after**2 + 0.5 * m2 * v2_after**2)) < 1e-10
        }
        
        return results
    
    def solve_hooke_wall(self, params):
        m = params['m1']
        k = params['k']
        v0 = params['v0']
        
        omega = np.sqrt(k / m)
        t_c = np.pi / omega
        t = np.linspace(0, 1.5 * t_c, 500)

        x = np.where(t <= t_c, (v0 / omega) * np.sin(omega * t), 0)
        v = np.where(t <= t_c, v0 * np.cos(omega * t), -v0)
        F = np.where(t <= t_c, -k * x, 0)

        results = {
            'time': t,
            'deformation': x,
            'velocity': v,
            'force': F,
            'contact_time': t_c,
            'max_deformation': np.max(x),
            'max_force': np.max(F)
        }
        
        return results
    
    def solve_hooke_balls(self, params):
        m1 = params['m1']
        m2 = params['m2']
        k = params['k']
        v1_initial = params['v0']
        v2_initial = params['u0']

        mu = m1 * m2 / (m1 + m2)
        v_rel_initial = v1_initial - v2_initial
        
        omega = np.sqrt(k / mu)
        t_c = np.pi / omega
        t = np.linspace(0, 1.5 * t_c, 500)

        x_rel = np.where(t <= t_c, (v_rel_initial / omega) * np.sin(omega * t), 0)
        v_rel = np.where(t <= t_c, v_rel_initial * np.cos(omega * t), -v_rel_initial)
        F = np.where(t <= t_c, -k * x_rel, 0)

        v_cm = (m1 * v1_initial + m2 * v2_initial) / (m1 + m2)
        v1 = v_cm + (m2 / (m1 + m2)) * v_rel
        v2 = v_cm - (m1 / (m1 + m2)) * v_rel

        results = {
            'time': t,
            'deformation': x_rel,
            'velocity': v_rel,
            'force': F,
            'v1': v1,
            'v2': v2,
            'v1_after': v1[-1],
            'v2_after': v2[-1],
            'contact_time': t_c,
            'max_deformation': np.max(x_rel),
            'max_force': np.max(F)
        }
        
        return results