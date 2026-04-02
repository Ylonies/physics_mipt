"""RK4 for dr/dt=v, dv/dt = (q/m) v×B."""

import numpy as np


def lorentz_rhs(state: np.ndarray, q_over_m: float, b_fn) -> np.ndarray:
    r, v = state[0:3], state[3:6]
    b = b_fn(r)
    a = q_over_m * np.cross(v, b)
    return np.concatenate([v, a])


def rk4_step(state: np.ndarray, dt: float, q_over_m: float, b_fn) -> np.ndarray:
    k1 = lorentz_rhs(state, q_over_m, b_fn)
    k2 = lorentz_rhs(state + 0.5 * dt * k1, q_over_m, b_fn)
    k3 = lorentz_rhs(state + 0.5 * dt * k2, q_over_m, b_fn)
    k4 = lorentz_rhs(state + dt * k3, q_over_m, b_fn)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    q_over_m: float,
    b_fn,
    *,
    escape_z: float | None = None,
) -> dict:
    """Returns history; stops after a step if |z| > escape_z."""
    state = np.concatenate([r0.astype(float), v0.astype(float)])
    hist_r = [state[0:3].copy()]
    hist_v = [state[3:6].copy()]
    times = [0.0]
    escaped_at = None
    if escape_z is not None and abs(state[2]) > escape_z:
        escaped_at = 0.0
    else:
        for k in range(n_steps):
            state = rk4_step(state, dt, q_over_m, b_fn)
            t = (k + 1) * dt
            hist_r.append(state[0:3].copy())
            hist_v.append(state[3:6].copy())
            times.append(t)
            if escape_z is not None and abs(state[2]) > escape_z:
                escaped_at = t
                break
    return {
        "t": np.array(times),
        "r": np.vstack(hist_r),
        "v": np.vstack(hist_v),
        "escaped_at": escaped_at,
    }
