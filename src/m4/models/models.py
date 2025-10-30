import math
import numpy as np


class PhysicsModels:
    def __init__(self, g: float = 9.81):
        self.g = g

    # --- 1) Inclined plane, pure rolling (no slip) ---
    def incline_no_slip(self, params):
        m = params['m']
        R = params['R']
        alpha = params['alpha']

        I = (2.0 / 5.0) * m * R * R
        a = self.g * math.sin(alpha) / (1.0 + I / (m * R * R))

        def model(t, state):
            s, v, omega = state
            ds_dt = v
            dv_dt = a
            domega_dt = a / R
            return [ds_dt, dv_dt, domega_dt]

        t_end = params['t_end']
        initial_state = [params['s0'], params['v0'], params['omega0']]
        t_span = (0.0, t_end)
        t_eval = np.linspace(0.0, t_end, 2000)
        return model, "Наклон без проскальзывания", initial_state, t_span, t_eval

    # --- 1b) Inclined plane with possible slip (Coulomb friction) ---
    def incline_with_slip(self, params):
        m = params['m']
        R = params['R']
        mu = params['mu']
        alpha = params['alpha']
        g = self.g
        I = (2.0 / 5.0) * m * R * R

        # helper: sign with zero -> 0
        def sgn(x: float) -> float:
            if x > 0:
                return 1.0
            if x < 0:
                return -1.0
            return 0.0

        def model(t, state):
            s, v, omega = state

            # Required friction for sticking (no slip) regime
            a_ns = g * math.sin(alpha) / (1.0 + I / (m * R * R))
            F_req = m * (g * math.sin(alpha) - a_ns)
            F_max = mu * m * g * math.cos(alpha)

            if abs(v - omega * R) < 1e-10 and abs(F_req) <= F_max:
                a = a_ns
                alpha_dot = a / R
                F = F_req
            else:
                F = - F_max * sgn(v - omega * R)  # kinetic friction
                a = g * math.sin(alpha) + F / m
                alpha_dot = - (F * R) / I

            ds_dt = v
            dv_dt = a
            domega_dt = alpha_dot
            return [ds_dt, dv_dt, domega_dt]

        t_end = params['t_end']
        initial_state = [params['s0'], params['v0'], params['omega0']]
        t_span = (0.0, t_end)
        t_eval = np.linspace(0.0, t_end, 4000)
        return model, "Наклон с возможным проскальзыванием", initial_state, t_span, t_eval

    # --- 2) Arbitrary rolling on a horizontal plane (vector form) ---
    def horizontal_general(self, params):
        m = params['m']
        R = params['R']
        mu = params['mu']
        g = self.g
        I = (2.0 / 5.0) * m * R * R
        Fx = params['Fx']
        Fy = params['Fy']

        ez = np.array([0.0, 0.0, 1.0])

        def model(t, state):
            x, y, vx, vy, wx, wy = state
            v = np.array([vx, vy])
            w = np.array([wx, wy, 0.0])

            # External in-plane force
            F_ext = np.array([Fx, Fy])

            # relative velocity at contact: v - R (omega × ez)
            omega_cross_ez = np.cross(w, ez)  # 3D
            v_rel = v - R * omega_cross_ez[:2]

            F_max = mu * m * g

            if np.linalg.norm(v_rel) < 1e-10:
                # candidate static friction that enforces no slip
                F_static = - (I / (I + m * R * R)) * F_ext
                if np.linalg.norm(F_static) <= F_max:
                    F_tan = F_static
                    a_vec = (F_ext + F_tan) / m
                    M_vec = -R * np.cross(ez, np.array([F_tan[0], F_tan[1], 0.0]))
                    w_dot = M_vec / I
                else:
                    # slips in direction of needed friction (which equals F_static direction)
                    dir_vec = F_static
                    if np.linalg.norm(dir_vec) < 1e-14:
                        dir_vec = np.array([0.0, 0.0])
                    else:
                        dir_vec = dir_vec / np.linalg.norm(dir_vec)
                    F_tan = -F_max * dir_vec
                    a_vec = (F_ext + F_tan) / m
                    M_vec = -R * np.cross(ez, np.array([F_tan[0], F_tan[1], 0.0]))
                    w_dot = M_vec / I
            else:
                dir_vec = v_rel / np.linalg.norm(v_rel)
                F_tan = -F_max * dir_vec
                a_vec = (F_ext + F_tan) / m
                M_vec = -R * np.cross(ez, np.array([F_tan[0], F_tan[1], 0.0]))
                w_dot = M_vec / I

            dx_dt = vx
            dy_dt = vy
            dvx_dt = a_vec[0]
            dvy_dt = a_vec[1]
            dwx_dt = w_dot[0]
            dwy_dt = w_dot[1]
            return [dx_dt, dy_dt, dvx_dt, dvy_dt, dwx_dt, dwy_dt]

        t_end = params['t_end']
        initial_state = [
            params['x0'], params['y0'],
            params['vx0'], params['vy0'],
            params['omega_x0'], params['omega_y0'],
        ]
        t_span = (0.0, t_end)
        t_eval = np.linspace(0.0, t_end, 3000)
        return model, "Горизонтальная плоскость (произвольное качение)", initial_state, t_span, t_eval

    def get_model(self, params):
        choice = params['choice']
        if choice == '1':
            return self.incline_no_slip(params)
        if choice == '1b':
            return self.incline_with_slip(params)
        return self.horizontal_general(params)


