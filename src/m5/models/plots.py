import matplotlib.pyplot as plt

class ResultVisualizer:
    @staticmethod
    def plot(results):
        t = results['time']
        theta = results['theta']
        omega = results['omega']
        K = results['K']
        V = results['V']

        E = results['E']

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(t, theta, label='θ(t)')
        plt.plot(t, omega, label='ω(t)')
        plt.xlabel('t, c')
        plt.legend()
        plt.title('Угловое отклонение и скорость')

        plt.subplot(1, 2, 2)
        plt.plot(t, K, label='K (кинетическая)')
        plt.plot(t, V, label='V (потенциальная)')
        plt.plot(t, E, label='E (полная)', linewidth=2)
        plt.xlabel('t, c')
        plt.ylabel('Энергия, Дж')
        plt.title('Энергия маятника (симплектический метод)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
