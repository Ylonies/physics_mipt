import numpy as np

from .models import InputHandler, PhysicsModels, ResultAnalyzer, ResultVisualizer, SimulationConfig

def main():
    print("Молекулярная динамика. М7А — адиабатический процесс под поршнем.")
    
    params = InputHandler.get_parameters()

    model = PhysicsModels()

    results = model.run(params)

    print("\n=== Результаты симуляции ===")
    cfg = SimulationConfig
    temps = results["temps"]
    pressures = results["pressures"]
    times = results["times"]

    tail = min(50, len(temps)) if len(temps) else 0
    T_mean = float(np.mean(temps[-tail:])) if tail else float("nan")
    P_mean = float(np.mean(pressures[-tail:])) if tail else float("nan")

    print(f"Конечное положение поршня: x_p = {results['x_p_final']:.4f}")
    print(f"Конечная скорость поршня: v_p = {results['v_p_final']:.4f}")
    if results["mode"] != 0:
        V_mean = float(np.mean(results["piston_xs"][-tail:]) * cfg.Ly) if tail else float("nan")
        print(f"Средний объём (среднее по последним {tail} измерениям, конец моделирования): V ≈ {V_mean:.6f}")
    print(f"Средняя температура (среднее по последним {tail} измерениям, конец моделирования): T ≈ {T_mean:.6f}")
    print(f"Среднее давление на поршень (среднее по последним {tail} измерениям, конец моделирования): P ≈ {P_mean:.6f}")
    print(f"Диагностических точек: {len(times)}")

    if results["mode"] == 3:
        t0 = results["t_event"]
        est = ResultAnalyzer.estimate_sound_speed_from_density(
            density_x=results["density_x"],
            density_profiles=results["density_profiles"],
            times=results["times"],
            t0=t0,
            piston_xs=results["piston_xs"],
        )
        c_est = est.get("c", float("nan"))
        c_th = ResultAnalyzer.theoretical_sound_speed(T_mean)
        print(f"\nСкорость звука (оценка по фронту плотности): c ≈ {c_est:.4f}")
        print(f"Скорость звука (теория, c = sqrt(gamma kB T / m)): c_th ≈ {c_th:.4f} (gamma={cfg.gamma:.3f})")

        V0 = cfg.Lx * cfg.Ly
        P_ext = params.get("P_ext_final", float("nan"))
        th_state = ResultAnalyzer.theoretical_final_state_pressure_step(
            N=params["N"], T0=params["T0"], V0=V0, P_ext=P_ext
        )
        print(f"\nТеория (адиабатический процесс против постоянного P_ext):")
        print(f"  V_f(theory) ≈ {th_state['V_f']:.6f}")
        print(f"  T_f(theory) ≈ {th_state['T_f']:.6f}")

    ResultVisualizer.plot_basic(results)

if __name__ == "__main__":
    main()
