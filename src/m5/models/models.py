import numpy as np


class PhysicsModels:
    @staticmethod
    def diode_current(u_v: float, params: dict) -> float:
        if u_v <= 0.0:
            return 0.0
        a = params["a_a_per_v3"]
        b = params["b_a_per_v2"]
        c = params["c_a_per_v"]
        return a * u_v**3 + b * u_v**2 + c * u_v

    def _tunnel_diode_generator(self, params: dict) -> dict:
        epsilon_v = params["epsilon_v"]
        r_ohm = params["r_ohm"]
        l_h = params["l_h"]
        c_f = params["c_f"]

        t = np.linspace(0.0, params["t_end_s"], params["n_points"])
        dt = float(t[1] - t[0])

        u = np.zeros_like(t)  # напряжение на диоде/конденсаторе
        i = np.zeros_like(t)  # ток через индуктивность
        id_arr = np.zeros_like(t)  # ток диода

        u[0] = params["u0_v"]
        i[0] = params["i0_a"]
        id_arr[0] = self.diode_current(u[0], params)

        for k in range(t.size - 1):
            id_k = self.diode_current(u[k], params)
            id_arr[k] = id_k

            # КЛЛ/КЗС для схемы:
            # L di/dt = epsilon - R i - u
            # C du/dt = i - I_diode(u)
            di_dt = (epsilon_v - r_ohm * i[k] - u[k]) / l_h
            du_dt = (i[k] - id_k) / c_f

            i[k + 1] = i[k] + dt * di_dt
            u[k + 1] = u[k] + dt * du_dt

        id_arr[-1] = self.diode_current(u[-1], params)

        return {
            "time": t,
            "u_diode_v": u,
            "i_inductor_a": i,
            "i_diode_a": id_arr,
        }

    def get_model(self, params):
        _ = params
        return self._tunnel_diode_generator, "RLC-генератор с туннельным диодом"
