import numpy as np
from .config import SimulationConfig

class ResultAnalyzer:
    @staticmethod
    def compute_density_profile(pos, nbins=50):
        cfg = SimulationConfig
        hist, edges = np.histogram(pos[:,0], bins=nbins, range=(0, cfg.Lx))
        bin_w = edges[1] - edges[0]
        density = hist / (bin_w * cfg.Ly)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, density

    @staticmethod
    def theoretical_sound_speed(T, *, gamma=None, kB=None, m=None):
        """
        Ideal-gas adiabatic sound speed: c = sqrt(gamma * kB * T / m).
        In this project the simulation uses dimensionless units by default.
        """
        cfg = SimulationConfig
        if gamma is None:
            gamma = cfg.gamma
        if kB is None:
            kB = cfg.kB
        if m is None:
            m = cfg.m
        T = float(T)
        if T <= 0:
            return np.nan
        return float(np.sqrt(gamma * kB * T / m))

    @staticmethod
    def theoretical_final_state_pressure_step(
        *, N, T0, V0, P_ext, dof=None, kB=None
    ):
        """
        Irreversible adiabatic compression/expansion against constant external pressure.

        Assumptions:
        - Q = 0
        - External pressure P_ext is constant after the step
        - Final equilibrium satisfies ideal gas: P_ext * V_f = N kB T_f
        - Internal energy: U = (dof/2) N kB T

        From (dof/2) N kB (T_f - T0) = -P_ext (V_f - V0) we get:
            V_f = [P_ext * V0 + (dof/2) N kB T0] / [P_ext * (1 + dof/2)]
            T_f = P_ext * V_f / (N kB)
        """
        cfg = SimulationConfig
        if dof is None:
            dof = cfg.dof
        if kB is None:
            kB = cfg.kB

        N = float(N)
        T0 = float(T0)
        V0 = float(V0)
        P_ext = float(P_ext)
        if N <= 0 or V0 <= 0 or kB <= 0 or P_ext <= 0:
            return {"V_f": np.nan, "T_f": np.nan}

        a = 0.5 * float(dof)
        V_f = (P_ext * V0 + a * N * kB * T0) / (P_ext * (1.0 + a))
        T_f = (P_ext * V_f) / (N * kB)
        return {"V_f": float(V_f), "T_f": float(T_f)}

    @staticmethod
    def estimate_sound_speed_from_density(
        *, density_x, density_profiles, times, t0, piston_xs=None, min_points=10, max_fit_points=40
    ):
        """
        Estimate sound speed from the motion of a density-gradient feature inside the gas.

        Steps:
        - baseline = mean of several profiles right before the event time t0
        - delta = profile - baseline
        - for each t >= t0, find the strongest |d/dx delta| away from the moving piston boundary
        - fit x_front(t) with a line over the early-time segment
        """
        if density_x is None or density_profiles is None:
            return {"c": np.nan, "fit_points": 0}

        x = np.asarray(density_x, dtype=float)
        prof = np.asarray(density_profiles, dtype=float)
        t = np.asarray(times, dtype=float)
        if prof.ndim != 2 or len(x) != prof.shape[1] or len(t) != prof.shape[0]:
            return {"c": np.nan, "fit_points": 0}

        if len(t) < 5:
            return {"c": np.nan, "fit_points": 0}

        dx = float(np.mean(np.diff(x)))
        if dx <= 0:
            return {"c": np.nan, "fit_points": 0}

        # reference: state right before the disturbance
        idx0 = int(np.searchsorted(t, float(t0), side="left"))
        if idx0 <= 0:
            baseline = prof[0]
        else:
            start = max(0, idx0 - 5)
            baseline = np.nanmean(prof[start:idx0], axis=0)
        delta = prof - baseline[None, :]

        # simple smoothing to reduce noise
        if delta.shape[1] >= 5:
            kernel = np.array([1, 2, 3, 2, 1], dtype=float)
            kernel /= kernel.sum()
            delta_s = np.empty_like(delta)
            for i in range(delta.shape[0]):
                delta_s[i] = np.convolve(delta[i], kernel, mode="same")
        else:
            delta_s = delta

        grad = np.diff(delta_s, axis=1) / dx
        x_mid = 0.5 * (x[:-1] + x[1:])

        # use only times after t0
        mask = t >= float(t0)
        if not np.any(mask):
            return {"c": np.nan, "fit_points": 0}

        idxs = np.where(mask)[0]
        piston = None if piston_xs is None else np.asarray(piston_xs, dtype=float)
        if piston is not None and piston.shape[0] != t.shape[0]:
            piston = None

        ignore_piston_bins = 6
        ignore_wall_bins = 2

        x_front = np.empty(len(idxs), dtype=float)
        t_front = t[idxs]
        for k, i in enumerate(idxs):
            valid = np.ones(grad.shape[1], dtype=bool)
            # ignore the left wall and near-piston discontinuity (vacuum region)
            valid &= x_mid >= ignore_wall_bins * dx
            if piston is not None:
                x_cut = piston[i] - ignore_piston_bins * dx
                valid &= x_mid <= x_cut
            if not np.any(valid):
                x_front[k] = np.nan
                continue
            gi = np.abs(grad[i])
            gi = np.where(valid, gi, -np.inf)
            j = int(np.nanargmax(gi))
            x_front[k] = x_mid[j]

        # keep only the first non-NaN points and fit early-time slope
        good = np.isfinite(x_front)
        if not np.any(good):
            return {"c": np.nan, "fit_points": 0}

        keep = np.where(good)[0][: int(max_fit_points)]
        if keep.size < min_points:
            return {"c": np.nan, "fit_points": int(keep.size)}

        tt = t_front[keep] - t_front[keep][0]
        xx = x_front[keep]
        # least squares line fit: x = a + b t
        b, a = np.polyfit(tt, xx, 1)
        c = float(abs(b))
        if c <= 0:
            return {"c": np.nan, "fit_points": int(keep.size)}
        return {"c": float(c), "fit_points": int(keep.size), "x0": float(a), "t0": float(t_front[keep][0])}