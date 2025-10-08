import matplotlib.pyplot as plt
import numpy as np

class ResultVisualizer:
    
    @staticmethod
    def plot(results, params, model_name):
        if params['model_choice'] == '2':
            if params['object_choice'] == '1':
                ResultVisualizer._plot_hooke_wall(results, params, model_name)
            else:
                ResultVisualizer._plot_hooke_balls(results, params, model_name)
        else:
            if params['object_choice'] == '1':
                ResultVisualizer._plot_elastic_wall(results, params, model_name)
            else:
                ResultVisualizer._plot_elastic_balls(results, params, model_name)
    
    @staticmethod
    def _plot_hooke_wall(results, params, model_name):
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(results['time'], results['deformation'], 'b-', linewidth=2)
        plt.ylabel("Сжатие x(t), м")
        plt.xlabel("Время t, с")
        plt.grid(True, alpha=0.3)
        plt.title("Деформация шара")

        plt.subplot(2, 2, 2)
        plt.plot(results['time'], results['velocity'], 'r-', linewidth=2)
        plt.ylabel("Скорость v(t), м/с")
        plt.xlabel("Время t, с")
        plt.grid(True, alpha=0.3)
        plt.title("Скорость шара")

        plt.subplot(2, 2, 3)
        plt.plot(results['time'], results['force'], 'g-', linewidth=2)
        plt.ylabel("Сила F(t), Н")
        plt.xlabel("Время t, с")
        plt.grid(True, alpha=0.3)
        plt.title("Сила взаимодействия")

        plt.subplot(2, 2, 4)
        ResultVisualizer._plot_trajectory_hooke_wall(results, params)
        
        plt.suptitle(f"{model_name}\n"
                    f"Масса: {params['m1']:.2f} кг, Начальная скорость: {params['v0']:.2f} м/с\n"
                    f"Время контакта: {results['contact_time']:.4f} с")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_trajectory_hooke_wall(results, params):
        t_before = np.linspace(-0.05, 0, 100)
        t_after = np.linspace(results['contact_time'], results['contact_time'] + 0.05, 100)
        
        x_before = params['v0'] * t_before
        y_before = np.zeros_like(x_before)
        
        t_contact = results['time'][results['time'] <= results['contact_time']]
        x_contact = np.zeros_like(t_contact)
        y_contact = results['deformation'][:len(t_contact)]
        
        x_after = -params['v0'] * (t_after - results['contact_time'])
        y_after = np.zeros_like(x_after)
        
        plt.plot(x_before, y_before, 'b-', linewidth=2, label='До удара')
        plt.plot(x_contact, y_contact, 'r-', linewidth=2, label='Контакт')
        plt.plot(x_after, y_after, 'g-', linewidth=2, label='После удара')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7, linewidth=2, label='Стенка')
        
        plt.xlabel('Координата x, м')
        plt.ylabel('Деформация y, м')
        plt.title('Траектория и деформация')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def _plot_hooke_balls(results, params, model_name):
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(results['time'], results['deformation'], 'b-', linewidth=2)
        plt.ylabel("Деформация x(t), м")
        plt.xlabel("Время t, с")
        plt.grid(True, alpha=0.3)
        plt.title("Деформация при столкновении")

        plt.subplot(2, 2, 2)
        plt.plot(results['time'], results['velocity'], 'r-', linewidth=2)
        plt.ylabel("Относительная скорость v(t), м/с")
        plt.xlabel("Время t, с")
        plt.grid(True, alpha=0.3)
        plt.title("Относительная скорость")

        plt.subplot(2, 2, 3)
        plt.plot(results['time'], results['force'], 'g-', linewidth=2)
        plt.ylabel("Сила F(t), Н")
        plt.xlabel("Время t, с")
        plt.grid(True, alpha=0.3)
        plt.title("Сила взаимодействия")

        plt.subplot(2, 2, 4)
        ResultVisualizer._plot_trajectory_hooke_balls(results, params)
        
        plt.suptitle(f"{model_name}\n"
                    f"Массы: {params['m1']:.2f} кг и {params['m2']:.2f} кг\n"
                    f"Скорости: {params['v0']:.2f} м/с и {params['u0']:.2f} м/с")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_trajectory_hooke_balls(results, params):
        t = np.linspace(-0.1, 0.1, 200)
        contact_start = 0
        
        x1_before = 0.1 + params['v0'] * t[t < contact_start]
        x1_contact = np.zeros_like(results['time'])
        x1_after = results['v1_after'] * (t[t >= contact_start] - contact_start)
        
        x2_before = -0.1 + params['u0'] * t[t < contact_start]
        x2_contact = np.zeros_like(results['time']) - 0.1
        x2_after = -0.1 + results['v2_after'] * (t[t >= contact_start] - contact_start)
        
        plt.plot(t[t < contact_start], x1_before, 'b-', linewidth=2, label='Шар 1 до')
        plt.plot(t[t < contact_start], x2_before, 'r-', linewidth=2, label='Шар 2 до')
        plt.plot(results['time'], x1_contact, 'b--', linewidth=1, label='Шар 1 контакт')
        plt.plot(results['time'], x2_contact, 'r--', linewidth=1, label='Шар 2 контакт')
        plt.plot(t[t >= contact_start], x1_after, 'b:', linewidth=2, label='Шар 1 после')
        plt.plot(t[t >= contact_start], x2_after, 'r:', linewidth=2, label='Шар 2 после')
        
        plt.xlabel('Время t, с')
        plt.ylabel('Координата x, м')
        plt.title('Положения шаров')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def _plot_elastic_wall(results, params, model_name):
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        ResultVisualizer._plot_trajectory_elastic_wall(results, params)
        
        plt.subplot(2, 2, 2)
        ResultVisualizer._plot_velocity_components_elastic_wall(results, params)
        
        plt.subplot(2, 2, 3)
        ResultVisualizer._plot_velocity_vectors_elastic_wall(results, params)
        
        plt.subplot(2, 2, 4)
        ResultVisualizer._plot_energy_momentum_elastic_wall(results, params)
        
        plt.suptitle(f"{model_name}\n"
                    f"Масса: {params['m1']:.2f} кг, Скорость: {params['v0']:.2f} м/с, "
                    f"Угол: {np.degrees(params['alpha']):.1f}°")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_trajectory_elastic_wall(results, params):
        t = np.linspace(-0.1, 0.1, 100)
        alpha = params['alpha']
        v0 = params['v0']
        
        x_before = v0 * np.cos(alpha) * t[t < 0]
        y_before = v0 * np.sin(alpha) * t[t < 0]
        
        x_after = -results['vx_after'] * t[t >= 0]
        y_after = results['vy_after'] * t[t >= 0]
        
        plt.plot(x_before, y_before, 'b-', linewidth=2, label='До удара')
        plt.plot(x_after, y_after, 'r-', linewidth=2, label='После удара')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7, linewidth=2, label='Стенка')
        
        plt.xlabel('Координата x, м')
        plt.ylabel('Координата y, м')
        plt.title('Траектория движения')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

    @staticmethod
    def _plot_velocity_components_elastic_wall(results, params):
        t = np.linspace(-0.1, 0.1, 100)
        vx_before = np.full_like(t[t < 0], params['v0'] * np.cos(params['alpha']))
        vy_before = np.full_like(t[t < 0], params['v0'] * np.sin(params['alpha']))
        vx_after = np.full_like(t[t >= 0], results['vx_after'])
        vy_after = np.full_like(t[t >= 0], results['vy_after'])
        
        plt.plot(t[t < 0], vx_before, 'b-', linewidth=2, label='Vx до')
        plt.plot(t[t < 0], vy_before, 'g-', linewidth=2, label='Vy до')
        plt.plot(t[t >= 0], vx_after, 'b--', linewidth=2, label='Vx после')
        plt.plot(t[t >= 0], vy_after, 'g--', linewidth=2, label='Vy после')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.xlabel('Время t, с')
        plt.ylabel('Скорость, м/с')
        plt.title('Компоненты скорости')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def _plot_velocity_vectors_elastic_wall(results, params):
        v_before = params['v0']
        alpha = params['alpha']
        
        plt.quiver(0, 0, v_before * np.cos(alpha), v_before * np.sin(alpha), 
                  angles='xy', scale_units='xy', scale=1, color='blue', 
                  width=0.015, label='До удара')
        plt.quiver(0, 0, results['vx_after'], results['vy_after'], 
                  angles='xy', scale_units='xy', scale=1, color='red', 
                  width=0.015, label='После удара')
        
        plt.xlim(-v_before*1.2, v_before*1.2)
        plt.ylim(-v_before*0.1, v_before*1.2)
        plt.xlabel('Vx, м/с')
        plt.ylabel('Vy, м/с')
        plt.title('Векторы скорости')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

    @staticmethod
    def _plot_energy_momentum_elastic_wall(results, params):
        t = np.linspace(-0.1, 0.1, 100)
        energy = np.full_like(t, 0.5 * params['m1'] * params['v0']**2)
        momentum_x = np.full_like(t, params['m1'] * params['v0'] * np.cos(params['alpha']))
        momentum_x[t >= 0] = params['m1'] * results['vx_after']
        
        plt.plot(t, energy, 'g-', linewidth=2, label='Энергия')
        plt.plot(t, momentum_x, 'r-', linewidth=2, label='Импульс X')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.xlabel('Время t, с')
        plt.ylabel('Энергия (Дж) / Импульс (кг·м/с)')
        plt.title('Энергия и импульс')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def _plot_elastic_balls(results, params, model_name):
        plt.figure(figsize=(12, 10))
    
        plt.subplot(2, 2, 1)
        ResultVisualizer._plot_trajectory_elastic_balls(results, params)

        plt.subplot(2, 2, 2)
        ResultVisualizer._plot_velocities_elastic_balls(results, params)

        plt.subplot(2, 2, 3)
        ResultVisualizer._plot_momentum_elastic_balls(results, params)
        
        plt.subplot(2, 2, 4)
        ResultVisualizer._plot_energy_elastic_balls(results, params)
        
        plt.suptitle(f"{model_name}\n"
                    f"Массы: {params['m1']:.2f} кг и {params['m2']:.2f} кг\n"
                    f"Скорости до: {params['v0']:.2f} м/с и {params['u0']:.2f} м/с")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_trajectory_elastic_balls(results, params):
        t = np.linspace(-0.1, 0.1, 100)
        
        x1_before = 0.1 + params['v0'] * t[t < 0]
        x1_after = results['v1_after'] * t[t >= 0]
        
        x2_before = -0.1 + params['u0'] * t[t < 0]
        x2_after = -0.1 + results['v2_after'] * t[t >= 0]
        
        plt.plot(t[t < 0], x1_before, 'b-', linewidth=2, label='Шар 1 до')
        plt.plot(t[t < 0], x2_before, 'r-', linewidth=2, label='Шар 2 до')
        plt.plot(t[t >= 0], x1_after, 'b--', linewidth=2, label='Шар 1 после')
        plt.plot(t[t >= 0], x2_after, 'r--', linewidth=2, label='Шар 2 после')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Момент удара')
        
        plt.xlabel('Время t, с')
        plt.ylabel('Координата x, м')
        plt.title('Положения шаров')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def _plot_velocities_elastic_balls(results, params):
        t = np.linspace(-0.1, 0.1, 100)
        
        v1 = np.concatenate([
            np.full_like(t[t < 0], params['v0']),
            np.full_like(t[t >= 0], results['v1_after'])
        ])
        v2 = np.concatenate([
            np.full_like(t[t < 0], params['u0']),
            np.full_like(t[t >= 0], results['v2_after'])
        ])
        
        plt.plot(t, v1, 'b-', linewidth=2, label='Шар 1')
        plt.plot(t, v2, 'r-', linewidth=2, label='Шар 2')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.xlabel('Время t, с')
        plt.ylabel('Скорость, м/с')
        plt.title('Скорости шаров')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def _plot_momentum_elastic_balls(results, params):
        t = np.linspace(-0.1, 0.1, 100)
        
        momentum_before = params['m1'] * params['v0'] + params['m2'] * params['u0']
        momentum_after = params['m1'] * results['v1_after'] + params['m2'] * results['v2_after']
        
        momentum = np.concatenate([
            np.full_like(t[t < 0], momentum_before),
            np.full_like(t[t >= 0], momentum_after)
        ])
        
        plt.plot(t, momentum, 'g-', linewidth=2)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=momentum_before, color='r', linestyle='--', alpha=0.7, label='Сохранение')
        
        plt.xlabel('Время t, с')
        plt.ylabel('Импульс, кг·м/с')
        plt.title('Импульс системы')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def _plot_energy_elastic_balls(results, params):
        t = np.linspace(-0.1, 0.1, 100)
        
        energy_before = 0.5 * params['m1'] * params['v0']**2 + 0.5 * params['m2'] * params['u0']**2
        energy_after = 0.5 * params['m1'] * results['v1_after']**2 + 0.5 * params['m2'] * results['v2_after']**2
        
        energy = np.concatenate([
            np.full_like(t[t < 0], energy_before),
            np.full_like(t[t >= 0], energy_after)
        ])
        
        plt.plot(t, energy, 'purple', linewidth=2)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=energy_before, color='r', linestyle='--', alpha=0.7, label='Сохранение')
        
        plt.xlabel('Время t, с')
        plt.ylabel('Энергия, Дж')
        plt.title('Энергия системы')
        plt.legend()
        plt.grid(True, alpha=0.3)