# models.py
from numba import njit, prange
import numpy as np
import math
from .config import SimulationConfig

class PhysicsModels:

    def __init__(self):
        pass

    def init_system(self, N, T0):
        cfg = SimulationConfig
        rng = np.random.default_rng(42)

        nx = int(math.sqrt(N * cfg.Lx / cfg.Ly))
        ny = int(math.ceil(N / nx))
        xs = np.linspace(cfg.radius, cfg.Lx - cfg.radius, nx)
        ys = np.linspace(cfg.radius, cfg.Ly - cfg.radius, ny)

        pos = np.zeros((N, 2), dtype=np.float64)
        idx = 0
        for y in ys:
            for x in xs:
                if idx >= N:
                    break
                jitter = (rng.random(2) - 0.5) * cfg.radius * 0.1
                pos[idx, 0] = x + jitter[0]
                pos[idx, 1] = y + jitter[1]
                idx += 1
            if idx >= N:
                break

        sigma = math.sqrt(cfg.kB * T0 / cfg.m) if T0 > 0 else 0.0
        vel = rng.normal(0, sigma, size=(N, 2))
        vel -= vel.mean(axis=0)

        x_p = cfg.Lx
        v_p = 0.0

        ncx = max(1, int(cfg.Lx / cfg.cell_size))
        ncy = max(1, int(cfg.Ly / cfg.cell_size))
        cell_w = cfg.Lx / ncx
        cell_h = cfg.Ly / ncy

        return pos, vel, x_p, v_p, ncx, ncy, cell_w, cell_h

    def run(self, params):
        """
        params contains:
        - N
        - T0
        - piston_target
        Runs simulation and returns diagnostics with averaged pressure for plotting.
        """
        N = params["N"]
        T0 = params["T0"]
        piston_target = params["piston_target"]

        pos, vel, x_p, v_p, ncx, ncy, cell_w, cell_h = self.init_system(N, T0)

        dt = SimulationConfig.dt
        max_steps = 300000
        diag_interval = 10

        out = self._run_numba_core(
            pos.copy(), vel.copy(), x_p, v_p,
            N, dt, max_steps, piston_target,
            SimulationConfig.radius, SimulationConfig.m, SimulationConfig.Mp,
            SimulationConfig.kB, SimulationConfig.Ly,
            ncx, ncy, cell_w, cell_h,
            diag_interval
        )

        # Unpack
        (pos_end, vel_end,
        temps_all, pressures_all, times_all, piston_xs_all,
        pos_history, piston_x_history,
        x_p_final, v_p_final, diag_count, steps_done) = out

        # Slice arrays to actual length
        temps = temps_all[:diag_count].copy()
        pressures = pressures_all[:diag_count].copy()
        times = times_all[:diag_count].copy()
        piston_xs = piston_xs_all[:diag_count].copy()

        # --- усреднение давления по 1 секунде ---
        interval_steps = int(1.0 / dt)  # сколько шагов = 1 секунда
        n_avg = len(pressures) // interval_steps + 1

        pressures_avg = np.zeros(n_avg)
        times_avg = np.zeros(n_avg)
        for i in range(n_avg):
            start = i * interval_steps
            end = min((i + 1) * interval_steps, len(pressures))
            pressures_avg[i] = pressures[start:end].mean()
            times_avg[i] = times[start:end].mean()

        results = {
            "pos": pos_end,
            "vel": vel_end,
            "temps": temps,
            "pressures": pressures,           # для диагностики
            "times": times,
            "pressures_avg": pressures_avg,   # для графика
            "times_avg": times_avg,           # для графика
            "piston_xs": piston_xs,
            "pos_history": pos_history,
            "piston_x_history": piston_x_history,
            "x_p_final": x_p_final,
            "v_p_final": v_p_final,
            "diag_count": diag_count,
            "steps_done": steps_done
        }

        return results


    @staticmethod
    @njit(parallel=True)
    def _run_numba_core(pos, vel, x_p, v_p,
                        N, dt, max_steps, piston_target,
                        radius, m, Mp, kB, Ly,
                        ncx, ncy, cell_w, cell_h,
                        diag_interval):
        """
        Numba core: kвазистатический поршень, cell-list столкновения, диагностика
        """

        # Diagnostics
        n_diag_max = max_steps // diag_interval + 2
        temps = np.zeros(n_diag_max, dtype=np.float64)
        pressures = np.zeros(n_diag_max, dtype=np.float64)
        times = np.zeros(n_diag_max, dtype=np.float64)
        piston_xs = np.zeros(n_diag_max, dtype=np.float64)

        # Snapshots: первый и последний
        pos_history = np.zeros((2, N, 2), dtype=np.float64)
        piston_x_history = np.zeros(2, dtype=np.float64)
        pos_history[0,:,:] = pos
        piston_x_history[0] = x_p

        # Cell-list
        avg_per_cell = N // (ncx * ncy)
        max_particles_per_cell = int(max(8, avg_per_cell*6 + 10))
        cells_idx = -np.ones((ncx, ncy, max_particles_per_cell), dtype=np.int64)
        cells_count = np.zeros((ncx, ncy), dtype=np.int64)

        diag_i = 0
        impulse_accum = 0.0
        steps_done = 0

        # Constants for kinematic piston
        kp = 50.0       # proportional gain for quasi-static piston
        damping = 10.0
        tol_x = 1e-6

        for step in range(max_steps):
            steps_done += 1

            # --- kinetic energy & v_rms ---
            KE = 0.0
            for i in range(N):
                KE += 0.5*m*(vel[i,0]**2 + vel[i,1]**2)
            T_inst = KE/(N*kB)
            if T_inst < 1e-12:
                T_inst = 1e-12
            v_rms = math.sqrt(2.0*kB*T_inst/m)

            # --- move piston (quasi-static) ---
            dx = piston_target - x_p
            if abs(dx) < tol_x:
                break
            F_piston = kp*dx - damping*v_p
            v_p += F_piston/Mp*dt
            x_p += v_p*dt
            if x_p < 2*radius:
                x_p = 2*radius
                v_p = 0.0

            # --- move particles ---
            for i in prange(N):
                pos[i,0] += vel[i,0]*dt
                pos[i,1] += vel[i,1]*dt

            # --- wall reflections ---
            for i in prange(N):
                if pos[i,0]-radius<0.0:
                    pos[i,0] = radius + (radius - pos[i,0])
                    vel[i,0] *= -1.0
                if pos[i,1]-radius<0.0:
                    pos[i,1] = radius + (radius - pos[i,1])
                    vel[i,1] *= -1.0
                if pos[i,1]+radius>Ly:
                    pos[i,1] = Ly-radius-(pos[i,1]+radius-Ly)
                    vel[i,1] *= -1.0

            # --- rebuild cell list ---
            for cx in range(ncx):
                for cy in range(ncy):
                    cells_count[cx,cy] = 0
            for i in range(N):
                cx = int(pos[i,0]/cell_w)
                cy = int(pos[i,1]/cell_h)
                if cx<0: cx=0
                if cy<0: cy=0
                if cx>=ncx: cx=ncx-1
                if cy>=ncy: cy=ncy-1
                c = cells_count[cx,cy]
                if c<max_particles_per_cell:
                    cells_idx[cx,cy,c] = i
                    cells_count[cx,cy] = c+1

            # --- collisions ---
            eps = 1e-10
            for cx in range(ncx):
                for cy in range(ncy):
                    nloc = cells_count[cx,cy]
                    for ii in range(nloc):
                        i = cells_idx[cx,cy,ii]
                        if i<0: continue
                        for dx_cell in (-1,0,1):
                            nx_cell = cx+dx_cell
                            if nx_cell<0 or nx_cell>=ncx: continue
                            for dy_cell in (-1,0,1):
                                ny_cell = cy+dy_cell
                                if ny_cell<0 or ny_cell>=ncy: continue
                                nloc2 = cells_count[nx_cell,ny_cell]
                                for jj in range(nloc2):
                                    j = cells_idx[nx_cell,ny_cell,jj]
                                    if j<0 or j<=i: continue
                                    dx_ = pos[j,0]-pos[i,0]
                                    dy_ = pos[j,1]-pos[i,1]
                                    dist2 = dx_**2 + dy_**2
                                    min2 = (2*radius)**2
                                    if dist2<min2:
                                        dist = math.sqrt(dist2) if dist2>0 else eps
                                        nxv = dx_/dist
                                        nyv = dy_/dist
                                        dvx = vel[j,0]-vel[i,0]
                                        dvy = vel[j,1]-vel[i,1]
                                        vn = dvx*nxv + dvy*nyv
                                        if vn<0.0:
                                            vi_n = vel[i,0]*nxv + vel[i,1]*nyv
                                            vj_n = vel[j,0]*nxv + vel[j,1]*nyv
                                            vel[i,0] += (vj_n-vi_n)*nxv
                                            vel[i,1] += (vj_n-vi_n)*nyv
                                            vel[j,0] += (vi_n-vj_n)*nxv
                                            vel[j,1] += (vi_n-vj_n)*nyv
                                        # overlap correction
                                        overlap = 2*radius - dist
                                        if overlap>0.0:
                                            corrx = 0.5*overlap*nxv
                                            corry = 0.5*overlap*nyv
                                            pos[i,0]-=corrx; pos[i,1]-=corry
                                            pos[j,0]+=corrx; pos[j,1]+=corry

            # --- piston collisions ---
            impulse_step = 0.0
            for i in range(N):
                if pos[i,0]+radius>x_p:
                    pos[i,0] = x_p-radius-(pos[i,0]+radius-x_p)
                    vi = vel[i,0]
                    vi_new = ((m-Mp)*vi+2*Mp*v_p)/(m+Mp)
                    vp_new = ((Mp-m)*v_p+2*m*vi)/(m+Mp)
                    impulse_step += abs(Mp*(vp_new-v_p))
                    vel[i,0] = vi_new

            impulse_accum += impulse_step

            # --- diagnostics ---
            if (step%diag_interval)==0:
                KE_tmp = 0.0
                for i in range(N):
                    KE_tmp += 0.5*m*(vel[i,0]**2+vel[i,1]**2)
                T_now = KE_tmp/(N*kB)
                temps[diag_i] = T_now
                pressures[diag_i] = impulse_accum/(Ly*diag_interval*dt)
                times[diag_i] = step*dt
                piston_xs[diag_i] = x_p
                diag_i += 1
                impulse_accum = 0.0

        # final snapshot
        pos_history[1,:,:] = pos
        piston_x_history[1] = x_p

        return (pos, vel,
                temps, pressures, times, piston_xs,
                pos_history, piston_x_history,
                x_p, v_p, diag_i, steps_done)
