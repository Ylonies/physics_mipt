from __future__ import annotations

import numpy as np


def _rk4_step(state: np.ndarray, t: float, dt: float, rhs) -> np.ndarray:
    k1 = rhs(t, state)
    k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = rhs(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_rc(u_in_fn, R: float, C: float, u_out0: float, t0: float, t_max: float, dt: float) -> dict:
    tau = R * C
    n = max(1, int(np.ceil((t_max - t0) / dt)))
    t = t0 + dt * np.arange(n + 1)
    u_out = np.empty(n + 1)
    u_out[0] = u_out0
    u_in = np.empty(n + 1)
    u_in[0] = u_in_fn(t0)

    def rhs(_t, y):
        return np.array([(u_in_fn(_t) - y[0]) / tau])

    for k in range(n):
        u_in[k + 1] = u_in_fn(t[k + 1])
        u_out[k + 1] = _rk4_step(np.array([u_out[k]]), t[k], dt, rhs)[0]
    return {"t": t, "u_in": u_in, "u_out": u_out}


def integrate_rlc(u_in_fn, R: float, L: float, C: float, i0: float, u_c0: float, t0: float, t_max: float, dt: float) -> dict:
    n = max(1, int(np.ceil((t_max - t0) / dt)))
    t = t0 + dt * np.arange(n + 1)
    state = np.array([i0, u_c0], dtype=float)
    hist_i = [i0]
    hist_uc = [u_c0]
    u_in = [u_in_fn(t0)]

    def rhs(_t, y):
        i, u_c = y[0], y[1]
        return np.array([(u_in_fn(_t) - R * i - u_c) / L, i / C])

    for k in range(n):
        state = _rk4_step(state, t[k], dt, rhs)
        hist_i.append(state[0])
        hist_uc.append(state[1])
        u_in.append(u_in_fn(t[k + 1]))
    i_arr = np.array(hist_i)
    return {"t": t, "u_in": np.array(u_in), "i": i_arr, "u_c": np.array(hist_uc), "u_out": R * i_arr}
