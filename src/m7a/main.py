from .models import InputHandler, PhysicsModels, ResultVisualizer

def main():
    print("Молекулярная динамика. М7А — адиабатический процесс под поршнем.")
    
    # 1. Получаем пользовательские параметры
    params = InputHandler.get_parameters()

    model = PhysicsModels()

    results = model.run(params)

    print("\n=== Результаты симуляции ===")
    print(f"Конечное положение поршня: x_p = {results['x_p_final']:.4f}")
    print(f"Конечная скорость поршня: v_p = {results['v_p_final']:.4f}")
    print(f"Средняя температура: {sum(results['temps'][-50:])/50:.4f}")
    print(f"Среднее давление: {sum(results['pressures'][-50:])/50:.4f}")
    print(f"Количество шагов: {len(results['times'])}")

    # 6. Визуализация
    ResultVisualizer.plot_basic(results)

if __name__ == "__main__":
    main()
