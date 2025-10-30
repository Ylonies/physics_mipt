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
        """
        Горизонтальная плоскость: шар катится под действием
        постоянной силы F по оси X. Унифицировано под формат [x, y, vx, vy, wx, wy].
        """

        m = params['m']
        R = params['R']
        mu = params['mu']
        F = params['F']  # постоянная горизонтальная сила (по оси X)
        g = self.g
        I = (2.0 / 5.0) * m * R * R
        F_max = mu * m * g

        ez = np.array([0.0, 0.0, 1.0])

        def model(t, state):
            x, y, vx, vy, wx, wy = state

            v = np.array([vx, vy])
            w = np.array([wx, wy, 0.0])
            F_ext = np.array([F, 0.0])  

            omega_cross_ez = np.cross(w, ez)
            v_rel = v - R * omega_cross_ez[:2]

            if np.linalg.norm(v_rel) < 1e-8:
                A = np.eye(2) / m + (R**2 / I) * np.array([[0, -1], [1, 0]])
                F_tan = -np.linalg.solve(A, F_ext)
                if np.linalg.norm(F_tan) > F_max:
                    F_tan = -F_max * F_tan / np.linalg.norm(F_tan)
            else:
                F_tan = -F_max * v_rel / np.linalg.norm(v_rel)

            a_vec = (F_ext + F_tan) / m
            M_vec = -R * np.cross(ez, np.array([F_tan[0], F_tan[1], 0.0]))
            w_dot = M_vec / I

            dx_dt, dy_dt = vx, vy
            dvx_dt, dvy_dt = a_vec
            dwx_dt, dwy_dt = w_dot[0], w_dot[1]

            return [dx_dt, dy_dt, dvx_dt, dvy_dt, dwx_dt, dwy_dt]

        x0 = params['x0']
        y0 = 0.0
        vx0 = params['v0']
        vy0 = 0.0
        wx0 = 0.0
        wy0 = params['omega0']
        t_end = params['t_end']

        initial_state = [x0, y0, vx0, vy0, wx0, wy0]
        t_span = (0.0, t_end)
        t_eval = np.linspace(0.0, t_end, 3000)

        title = "Горизонтальная плоскость: произвольное качение с постоянной силой"

        return model, title, initial_state, t_span, t_eval

    def get_model(self, params):
        choice = params['choice']
        if choice == '1':
            return self.incline_no_slip(params)
        if choice == '1b':
            return self.incline_with_slip(params)
        if choice == '2':
            return self.horizontal_general(params)


