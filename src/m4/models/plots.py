import numpy as np
import matplotlib.pyplot as plt


class ResultVisualizer:
    @staticmethod
    def plot(results, params, model_name):
        t = results['time']
        E = results.get('energy_total')

        plt.figure(figsize=(12, 10))

        # === (1,2): Траектория ===
        plt.subplot(2, 2, (1, 2))
        if 'Наклон' in model_name:
            s = results.get('s')
            alpha = params['alpha']
            x = s * np.cos(alpha)
            y = -s * np.sin(alpha)
            plt.plot(x, y, 'b-', linewidth=2, label='Траектория проекции')
            plt.xlabel('x (по горизонтали), м')
            plt.ylabel('y (вертикаль), м')
            plt.title(f'Качение по наклонной: {model_name}')
            plt.grid(True, alpha=0.3)
        else:
            x = results.get('x')
            y = results.get('y')
            plt.plot(x, y, 'b-', linewidth=2)
            plt.xlabel('x, м')
            plt.ylabel('y, м')
            plt.title(f'Качение по горизонтали')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)

        # === (3): Энергия ===
        plt.subplot(2, 2, 3)
        if E is not None:
            plt.plot(t, E, 'r-', linewidth=2)
            plt.title('Полная энергия')
            plt.xlabel('t, с')
            plt.ylabel('E, Дж')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Энергия не рассчитана', ha='center', va='center')

        # === (4): Скорости ===
        plt.subplot(2, 2, 4)
        R = params['R']

        if 'Наклон' in model_name:
            v = results.get('v')
            omega = results.get('omega')
            plt.plot(t, v, 'r-', linewidth=2, label='|v| (линейная)')
            plt.plot(t, np.abs(omega) * R, 'b--', linewidth=2, label='|ωR| (эквивалентная)')
            plt.title('Сравнение линейной и вращательной скоростей (наклон)')
            plt.xlabel('t, с')
            plt.ylabel('м/с')
            plt.legend()
            plt.grid(True, alpha=0.3)

        else:
            vx = results.get('vx')
            vy = results.get('vy')
            wx = results.get('wx')
            wy = results.get('wy')

            if vx is not None:
                v_mag = np.sqrt(vx**2 + vy**2)
                omega_mag = np.sqrt(wx**2 + wy**2)
                plt.plot(t, v_mag, 'r-', linewidth=2, label='|v|')
                plt.plot(t, omega_mag * R, 'b--', linewidth=2, label='R|ω|')
                plt.title('Сравнение скоростей: поступательной и вращательной (горизонталь)')
                plt.xlabel('t, с')
                plt.ylabel('м/с')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'Нет данных о скоростях', ha='center', va='center')

        plt.tight_layout()
        plt.show()
