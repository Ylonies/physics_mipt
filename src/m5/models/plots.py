import matplotlib.pyplot as plt

class ResultVisualizer:
    @staticmethod
    def plot(results):
        t = results['time']
        theta = results['theta']
        omega = results['omega']
        E = results['energy_total']

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(t, theta, 'b', label='θ(t)')
        plt.plot(t, omega, 'r', label='ω(t)')
        plt.xlabel('t, с')
        plt.ylabel('θ, ω')
        plt.legend()
        plt.title('Угловое отклонение и скорость')

        plt.subplot(1, 2, 2)
        plt.plot(t, E, 'g', label='E(t)')
        plt.xlabel('t, с')
        plt.ylabel('Энергия, Дж')
        plt.title('Полная энергия маятника')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
