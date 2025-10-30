import numpy as np
import matplotlib.pyplot as plt


class ResultVisualizer:
    @staticmethod
    def plot(results, params, model_name):
        t = results['time']
        E = results.get('energy_total')

        plt.figure(figsize=(12, 10))

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

        # Energy vs time
        plt.subplot(2, 2, 3)
        if E is not None:
            plt.plot(t, E, 'r-', linewidth=2)
            plt.title('Полная энергия')
            plt.xlabel('t, с')
            plt.ylabel('E, Дж')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Энергия не рассчитана', ha='center', va='center')

        # Slip indicator
        plt.subplot(2, 2, 4)
        R = params['R']
        if 'Наклон' in model_name:
            v = results.get('v')
            omega = results.get('omega')
            slip_measure = np.abs(v - omega * R)
            plt.plot(t, slip_measure, 'g-')
            plt.title('|v − ωR| (наклон)')
            plt.xlabel('t, с')
            plt.ylabel('м/с')
            plt.grid(True, alpha=0.3)
        else:
            vx = results.get('vx')
            vy = results.get('vy')
            wx = results.get('wx')
            wy = results.get('wy')
            if vx is not None:
                slip_measure = np.sqrt((vx + R * wy) ** 2 + (vy - R * wx) ** 2)
                plt.plot(t, slip_measure, 'g-')
                plt.title('‖v − R(ω × ez)‖ (горизонталь)')
                plt.xlabel('t, с')
                plt.ylabel('м/с')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'Нет данных о скоростях', ha='center', va='center')

        plt.tight_layout()
        plt.show()


