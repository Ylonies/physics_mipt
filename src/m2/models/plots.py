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
        plt.axvline(x=results['contact_time'], color='r', linestyle='--', alpha=0.7)
        plt.ylabel("Сжатие x(t), м")
        plt.xlabel("Время t, с")
        plt.grid(True, alpha=0.3)
        plt.title("Деформация шара")

        plt.subplot(2, 2, 2)
        plt.plot(results['time'], results['velocity'], 'r-', linewidth=2)
        plt.axvline(x=results['contact_time'], color='r', linestyle='--', alpha=0.7)
        plt.ylabel("Скорость v(t), м/с")
        plt.xlabel("Время t, с")
        plt.grid(True, alpha=0.3)
        plt.title("Скорость шара")

        plt.subplot(2, 2, 3)
        plt.plot(results['time'], results['force'], 'g-', linewidth=2)
        plt.axvline(x=results['contact_time'], color='r', linestyle='--', alpha=0.7)
        plt.ylabel("Сила F(t), Н")
        plt.xlabel("Время t, с")
        plt.grid(True, alpha=0.3)
        plt.title("Сила взаимодействия")

        plt.subplot(2, 2, 4)
        t_traj = np.linspace(-0.1, 0.2, 300)
        x_traj = np.where(t_traj < 0, params['v0'] * t_traj, 
                         -params['v0'] * (t_traj - 2 * results['contact_time']))
        
        plt.plot(t_traj, x_traj, 'purple', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=2, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        plt.axvline(x=results['contact_time'], color='r', linestyle='--', alpha=0.7)
        plt.xlabel("Время, с")
        plt.ylabel("Координата, м")
        plt.title("Траектория движения")
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"{model_name}")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _plot_hooke_balls(results, params, model_name):
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(results['time'], results['deformation'], 'b-', linewidth=2)
        plt.axvline(x=results['contact_time'], color='r', linestyle='--', alpha=0.7)
        plt.ylabel("Деформация, м")
        plt.xlabel("Время, с")
        plt.grid(True, alpha=0.3)
        plt.title("Деформация при контакте")

        plt.subplot(2, 2, 2)
        plt.plot(results['time'], results['force'], 'r-', linewidth=2)
        plt.axvline(x=results['contact_time'], color='r', linestyle='--', alpha=0.7)
        plt.ylabel("Сила, Н")
        plt.xlabel("Время, с")
        plt.grid(True, alpha=0.3)
        plt.title("Сила взаимодействия")

        plt.subplot(2, 2, 3)
        plt.plot(results['time'], results['v1'], 'b-', linewidth=2, label='Шар 1')
        plt.plot(results['time'], results['v2'], 'r-', linewidth=2, label='Шар 2')
        plt.axvline(x=results['contact_time'], color='k', linestyle='--', alpha=0.7)
        plt.ylabel("Скорость, м/с")
        plt.xlabel("Время, с")
        plt.title("Скорости шаров")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        energy1 = 0.5 * params['m1'] * results['v1']**2
        energy2 = 0.5 * params['m2'] * results['v2']**2
        total_energy = energy1 + energy2
        
        plt.plot(results['time'], energy1, 'b-', linewidth=2, label='Энергия шара 1')
        plt.plot(results['time'], energy2, 'r-', linewidth=2, label='Энергия шара 2')
        plt.plot(results['time'], total_energy, 'g--', linewidth=2, label='Общая энергия')
        plt.axvline(x=results['contact_time'], color='k', linestyle='--', alpha=0.7)
        plt.ylabel("Энергия, Дж")
        plt.xlabel("Время, с")
        plt.title("Кинетические энергии")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"{model_name}")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _plot_elastic_wall(results, params, model_name):
        plt.figure(figsize=(12, 10))
        
        t = np.linspace(-0.05, 0.05, 200)
        
        plt.subplot(2, 2, 1)
        x_before = params['v0'] * np.cos(params['alpha']) * t[t < 0]
        y_before = params['v0'] * np.sin(params['alpha']) * t[t < 0]
        x_after = results['vx_after'] * t[t >= 0]
        y_after = results['vy_after'] * t[t >= 0]
        
        plt.plot(x_before, y_before, 'b-', linewidth=2, label='До удара')
        plt.plot(x_after, y_after, 'r-', linewidth=2, label='После удара')
        plt.axvline(x=0, color='k', linestyle='-', linewidth=2, alpha=0.7, label='Стенка')
        plt.xlabel("x, м")
        plt.ylabel("y, м")
        plt.title("Траектория движения")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.subplot(2, 2, 2)
        vx = np.concatenate([np.full_like(t[t < 0], params['v0'] * np.cos(params['alpha'])),
                            np.full_like(t[t >= 0], results['vx_after'])])
        vy = np.concatenate([np.full_like(t[t < 0], params['v0'] * np.sin(params['alpha'])),
                            np.full_like(t[t >= 0], results['vy_after'])])
        
        plt.plot(t, vx, 'b-', linewidth=2, label='Vx')
        plt.plot(t, vy, 'r-', linewidth=2, label='Vy')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel("Время, с")
        plt.ylabel("Скорость, м/с")
        plt.title("Компоненты скорости")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        v_total = np.sqrt(vx**2 + vy**2)
        plt.plot(t, v_total, 'g-', linewidth=2)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel("Время, с")
        plt.ylabel("Скорость, м/с")
        plt.title("Модуль скорости")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        angle_before = np.full_like(t[t < 0], np.degrees(params['alpha']))
        angle_after = np.full_like(t[t >= 0], np.degrees(results['alpha_after']))
        
        plt.plot(t[t < 0], angle_before, 'b-', linewidth=2, label='Угол падения')
        plt.plot(t[t >= 0], angle_after, 'r-', linewidth=2, label='Угол отражения')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel("Время, с")
        plt.ylabel("Угол, °")
        plt.title("Углы движения")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"{model_name}")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _plot_elastic_balls(results, params, model_name):
        plt.figure(figsize=(12, 10))
        
        t = np.linspace(-0.05, 0.05, 200)
        
        plt.subplot(2, 2, 1)
        v1 = np.concatenate([np.full_like(t[t < 0], params['v0']),
                            np.full_like(t[t >= 0], results['v1_after'])])
        v2 = np.concatenate([np.full_like(t[t < 0], params['u0']),
                            np.full_like(t[t >= 0], results['v2_after'])])
        
        plt.plot(t, v1, 'b-', linewidth=2, label='Шар 1')
        plt.plot(t, v2, 'r-', linewidth=2, label='Шар 2')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel("Время, с")
        plt.ylabel("Скорость, м/с")
        plt.title("Скорости шаров")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        energy = 0.5 * params['m1'] * v1**2 + 0.5 * params['m2'] * v2**2
        plt.plot(t, energy, 'g-', linewidth=2)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel("Время, с")
        plt.ylabel("Энергия, Дж")
        plt.title("Суммарная энергия")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        momentum = params['m1'] * v1 + params['m2'] * v2
        plt.plot(t, momentum, 'purple', linewidth=2)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel("Время, с")
        plt.ylabel("Импульс, кг·м/с")
        plt.title("Суммарный импульс")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        v_rel = v1 - v2
        plt.plot(t, v_rel, 'orange', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel("Время, с")
        plt.ylabel("Отн. скорость, м/с")
        plt.title("Относительная скорость")
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"{model_name}")
        plt.tight_layout()
        plt.show()