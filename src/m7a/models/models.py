import math
from numba import njit
import numpy as np
from .config import SimulationConfig

@njit
def _move_particles(pos, vel, dt):
    for i in range(pos.shape[0]):
        pos[i,0] += vel[i,0]*dt
        pos[i,1] += vel[i,1]*dt

@njit
def _apply_walls(pos, vel, radius, Lx, Ly, m):
    impulse = 0.0
    N = pos.shape[0]
    for i in range(N):
        if pos[i,0]-radius<0:
            pos[i,0] = radius + (radius-pos[i,0])
            vel[i,0] = -vel[i,0]
            impulse += 2*m*abs(vel[i,0])
        if pos[i,0]+radius>Lx:
            pos[i,0] = Lx-radius - (pos[i,0]+radius-Lx)
            vel[i,0] = -vel[i,0]
            impulse += 2*m*abs(vel[i,0])
        if pos[i,1]-radius<0:
            pos[i,1] = radius + (radius-pos[i,1])
            vel[i,1] = -vel[i,1]
        if pos[i,1]+radius>Ly:
            pos[i,1] = Ly-radius - (pos[i,1]+radius-Ly)
            vel[i,1] = -vel[i,1]
    return impulse

@njit
def _build_cells(pos, ncx, ncy, cell_w, cell_h, max_particles_per_cell):
    cells_idx = -np.ones((ncx, ncy, max_particles_per_cell), np.int64)
    cells_count = np.zeros((ncx, ncy), np.int64)
    N = pos.shape[0]
    for i in range(N):
        cx = min(max(int(pos[i,0]/cell_w),0), ncx-1)
        cy = min(max(int(pos[i,1]/cell_h),0), ncy-1)
        c = cells_count[cx,cy]
        if c<max_particles_per_cell:
            cells_idx[cx,cy,c] = i
            cells_count[cx,cy] = c+1
    return cells_idx, cells_count

@njit
def _particle_collisions(pos, vel, radius, ncx, ncy, cells_idx, cells_count):
    eps = 1e-12
    N = pos.shape[0]
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
                                if vn<0:
                                    vi_n = vel[i,0]*nxv + vel[i,1]*nyv
                                    vj_n = vel[j,0]*nxv + vel[j,1]*nyv
                                    vel[i,0] += (vj_n-vi_n)*nxv
                                    vel[i,1] += (vj_n-vi_n)*nyv
                                    vel[j,0] += (vi_n-vj_n)*nxv
                                    vel[j,1] += (vi_n-vj_n)*nyv
                                overlap = 2*radius - dist
                                if overlap>0:
                                    corrx = 0.5*overlap*nxv
                                    corry = 0.5*overlap*nyv
                                    pos[i,0]-=corrx; pos[i,1]-=corry
                                    pos[j,0]+=corrx; pos[j,1]+=corry

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
        N = params["N"]
        T0 = params["T0"]
        piston_target = params.get("piston_target", None)
        mode = params["mode"]

        pos, vel, x_p, v_p, ncx, ncy, cell_w, cell_h = self.init_system(N, T0)
        dt = SimulationConfig.dt
        max_steps = 300000
        diag_interval = 10

        if mode == 0:
            out = PhysicsModels._run_fixed_volume(pos.copy(), vel.copy(), N, dt, max_steps,
                                                  SimulationConfig.radius, SimulationConfig.m,
                                                  SimulationConfig.kB, SimulationConfig.Lx, SimulationConfig.Ly,
                                                  ncx, ncy, cell_w, cell_h, diag_interval)
        else:
            fast = (mode==2)
            out = PhysicsModels._run_with_piston(pos.copy(), vel.copy(), x_p, v_p,
                                                 N, dt, max_steps, piston_target,
                                                 SimulationConfig.radius, SimulationConfig.m, SimulationConfig.Mp,
                                                 SimulationConfig.kB, SimulationConfig.Lx, SimulationConfig.Ly,
                                                 ncx, ncy, cell_w, cell_h,
                                                 diag_interval, fast=fast)
            
        
        (pos_end, vel_end, temps, pressures, times, piston_xs,
         pos_history, piston_x_history, snapshot_indices,
         x_p_final, v_p_final, diag_count, steps_done) = out

        if mode == 0:
            volumes = np.full_like(times, SimulationConfig.Lx * SimulationConfig.Ly, dtype=np.float64)
        else:
            volumes = piston_xs * SimulationConfig.Ly
        samples_per_second = max(1, int(1.0/(diag_interval*dt)))
        if len(pressures)==0:
            pressures_avg = np.array([], dtype=np.float64)
            times_avg = np.array([], dtype=np.float64)
        else:
            n_avg = (len(pressures)+samples_per_second-1)//samples_per_second
            pressures_avg = np.full(n_avg, np.nan, dtype=np.float64)
            times_avg = np.full(n_avg, np.nan, dtype=np.float64)
            for i in range(n_avg):
                start = i*samples_per_second
                end = min((i+1)*samples_per_second, len(pressures))
                if end>start:
                    pressures_avg[i] = pressures[start:end].mean()
                    times_avg[i] = times[start:end].mean()

        return {
            "pos": pos_end,
            "vel": vel_end,
            "temps": temps,
            "pressures": pressures,
            "times": times,
            "pressures_avg": pressures_avg,
            "times_avg": times_avg,
            "piston_xs": piston_xs,
            "pos_history": pos_history,
            "piston_x_history": piston_x_history,
            "snapshot_indices": snapshot_indices,
            "x_p_final": x_p_final,
            "v_p_final": v_p_final,
            "diag_count": diag_count,
            "steps_done": steps_done
        }

    @staticmethod
    @njit
    def _run_fixed_volume(pos, vel, N, dt, max_steps, radius, m, kB, Lx, Ly, ncx, ncy, cell_w, cell_h, diag_interval):
        n_diag_max = max_steps // diag_interval + 2
        temps = np.zeros(n_diag_max)
        pressures = np.zeros(n_diag_max)
        times = np.zeros(n_diag_max)
        piston_xs = np.zeros(n_diag_max)
        pos_history = np.zeros((2, N, 2))
        piston_x_history = np.zeros(2)
        snapshot_indices = np.zeros(2, dtype=np.int64)

        pos_history[0,:,:] = pos
        piston_x_history[0] = Lx
        snapshot_indices[0] = 0

        avg_per_cell = N // (ncx*ncy)
        max_particles_per_cell = int(max(8, avg_per_cell*6 + 10))
        diag_i = 0
        impulse_accum = 0.0
        steps_done = 0

        for step in range(max_steps):
            steps_done += 1
            _move_particles(pos, vel, dt)
            impulse_accum += _apply_walls(pos, vel, radius, Lx, Ly, m)
            cells_idx, cells_count = _build_cells(pos, ncx, ncy, cell_w, cell_h, max_particles_per_cell)
            _particle_collisions(pos, vel, radius, ncx, ncy, cells_idx, cells_count)

            if step % diag_interval == 0:
                KE = 0.0
                for i in range(N):
                    KE += 0.5*m*(vel[i,0]**2 + vel[i,1]**2)
                temps[diag_i] = KE/(N*kB)
                pressures[diag_i] = impulse_accum/(Ly*diag_interval*dt)
                times[diag_i] = step*dt
                piston_xs[diag_i] = Lx
                diag_i += 1
                impulse_accum = 0.0

        pos_history[1,:,:] = pos
        piston_x_history[1] = Lx
        snapshot_indices[1] = diag_i-1 if diag_i>0 else 0

        return pos, vel, temps[:diag_i], pressures[:diag_i], times[:diag_i], piston_xs[:diag_i], \
               pos_history, piston_x_history, snapshot_indices, Lx, 0.0, diag_i, steps_done

    @staticmethod
    @njit
    def _run_with_piston(pos, vel, x_p, v_p, N, dt, max_steps, piston_target,
                         radius, m, Mp, kB, Lx, Ly, ncx, ncy, cell_w, cell_h,
                         diag_interval, fast=False):
        n_diag_max = max_steps // diag_interval + 2
        temps = np.zeros(n_diag_max)
        pressures = np.zeros(n_diag_max)
        times = np.zeros(n_diag_max)
        piston_xs = np.zeros(n_diag_max)
        pos_history = np.zeros((2, N, 2))
        piston_x_history = np.zeros(2)
        snapshot_indices = np.zeros(2, dtype=np.int64)

        pos_history[0,:,:] = pos
        piston_x_history[0] = x_p
        snapshot_indices[0] = 0

        avg_per_cell = N // (ncx * ncy)
        max_particles_per_cell = int(max(8, avg_per_cell*6 + 10))

        diag_i = 0
        impulse_accum = 0.0
        steps_done = 0

        if fast:
            drive_speed = 1.0    
        else:
            drive_speed = 0.02  

        v_eq_thresh = 1e-2            # порог для максимальной скорости частиц
        consec_eq_required = 5        # количество подряд диагностик с условием равновесия
        eq_counter = 0
        piston_fixed = False

        for step in range(max_steps):
            steps_done += 1
            dx = piston_target - x_p

            # ---------- управление положением поршня ----------
            if not piston_fixed:
                if fast:
                    # быстрый режим: двигаем поршень с постоянной скоростью drive_speed в направлении цели
                    v_p = math.copysign(drive_speed, dx) if abs(dx) > 0 else 0.0
                    move = v_p * dt
                    # не пересекаем цель
                    if abs(move) > abs(dx):
                        move = dx
                    x_p += move
                else:
                    # квазистатический: плавное приближение к цели с ограничением по скорости
                    desired = math.copysign(min(abs(dx), drive_speed), dx) if abs(dx) > 0 else 0.0
                    # простая П-регуляция скорости (плавная)
                    # т.е. стремимся к desired за один маленький шаг (но не резко)
                    v_p = desired
                    move = v_p * dt
                    if abs(move) > abs(dx):
                        move = dx
                    x_p += move

                # если достигли цели (или очень близко) — фиксируем поршень
                if abs(piston_target - x_p) < 1e-6:
                    x_p = piston_target
                    v_p = 0.0
                    piston_fixed = True
                    # не обнуляем eq_counter - даём системе время утихнуть
                    eq_counter = 0

            # если поршень уже зафиксирован — x_p не меняется, v_p = 0
            # ---------- движение частиц ----------
            _move_particles(pos, vel, dt)

            # ---------- столкновения со стенами ----------
            impulse_accum += _apply_walls(pos, vel, radius, Lx, Ly, m)

            # ---------- cell list и межчастичные столкновения ----------
            cells_idx, cells_count = _build_cells(pos, ncx, ncy, cell_w, cell_h, max_particles_per_cell)
            _particle_collisions(pos, vel, radius, ncx, ncy, cells_idx, cells_count)

            # ---------- взаимодействие с поршнем ----------
            # ПОРШЕНЬ НЕ РЕАГИРУЕТ: при ударе частица отражается (спекулярно относительно движущейся границы),
            # а скорость поршня не обновляется.
            for i in range(N):
                if pos[i,0] + radius > x_p:
                    # позиционная коррекция: не пропускать частицу за поршень
                    pos[i,0] = x_p - radius
                    # относительная скорость по x в системе поршня
                    v_rel = vel[i,0] - v_p
                    # отражение: v_rel' = -v_rel  => v_new = -v_rel + v_p = -vel + 2*v_p
                    v_new = -vel[i,0] + 2.0 * v_p
                    # накопление импульса для расчёта давления (импульс, переданный стенке)
                    # импульс по модулю:  m * |v_new - vel_old|
                    impulse_accum += m * abs(v_new - vel[i,0])
                    vel[i,0] = v_new

            # ---------- диагностика ----------
            if step % diag_interval == 0:
                KE = 0.0
                vmax = 0.0
                for i in range(N):
                    KE += 0.5 * m * (vel[i,0]**2 + vel[i,1]**2)
                    vmag = math.sqrt(vel[i,0]**2 + vel[i,1]**2)
                    if vmag > vmax:
                        vmax = vmag
                temps[diag_i] = KE / (N * kB)
                pressures[diag_i] = impulse_accum / (Ly * diag_interval * dt) if diag_interval*dt > 0 else 0.0
                times[diag_i] = step * dt
                piston_xs[diag_i] = x_p
                diag_i += 1
                impulse_accum = 0.0

                # если поршень зафиксирован — проверяем критерий равновесия
                if piston_fixed:
                    if vmax < v_eq_thresh:
                        eq_counter += 1
                    else:
                        eq_counter = 0

                    # если несколько подряд диагностик показывают тихую систему — завершаем
                    if eq_counter >= consec_eq_required:
                        break

            # safety: если всё ещё не достиг цели и шаги закончились, цикл завершится по max_steps

        # финальная диагностика (если последний diag не записан)
        if diag_i == 0 or (diag_i > 0 and times[diag_i-1] < steps_done * dt - 1e-12):
            KE = 0.0
            for i in range(N):
                KE += 0.5 * m * (vel[i,0]**2 + vel[i,1]**2)
            temps[diag_i] = KE / (N * kB)
            pressures[diag_i] = 0.0
            times[diag_i] = steps_done * dt
            piston_xs[diag_i] = x_p
            diag_i += 1

        # конечный snapshot
        pos_history[1,:,:] = pos
        piston_x_history[1] = x_p
        snapshot_indices[1] = diag_i-1 if diag_i>0 else 0

        return pos, vel, temps[:diag_i], pressures[:diag_i], times[:diag_i], piston_xs[:diag_i], \
               pos_history, piston_x_history, snapshot_indices, x_p, v_p, diag_i, steps_done


