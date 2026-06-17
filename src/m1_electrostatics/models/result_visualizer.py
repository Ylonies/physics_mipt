import matplotlib.pyplot as plt
import numpy as np

class ResultVisualizer:
    def __init__(self, theme='seaborn-v0_8'):
        try:
            plt.style.use(theme)
        except:
            plt.style.use('ggplot')

    def plot_charge_distribution(self, ax, points, charges, areas, R):
        half_n = len(points) // 2
        pts = points[:half_n]
        sigma = charges[:half_n] / areas[:half_n]

        scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                             c=sigma, cmap='plasma', s=20, alpha=0.8)

        ax.set_title(f"Распределение плотности заряда (R={R}м)", fontsize=10, pad=20)

        ax.set_xlabel("X, м", labelpad=10)
        ax.set_ylabel("Y, м", labelpad=10)
        ax.set_zlabel("Z, м", labelpad=10)

        ax.view_init(elev=30, azim=45)
        return scatter


    def show_all_plots(self, points, charges, areas, R, d):
        """Метод-обертка с настройкой отступов"""
        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(121, projection='3d')
        scatter = self.plot_charge_distribution(ax1, points, charges, areas, R)

        cbar = fig.colorbar(scatter, ax=ax1, shrink=0.5, aspect=15, pad=0.1)
        cbar.set_label("$\sigma$ [Кл/м²]", rotation=270, labelpad=15)

        ax2 = fig.add_subplot(122)
        self.plot_electric_field(ax2, points, charges, R, d)

        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

        plt.show()


    def plot_electric_field(self, ax, points, charges, R, d):
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        z_center = (z_min + z_max) / 2
        half = len(charges) // 2
        q_bottom = np.sum(charges[:half])
        q_top = np.sum(charges[half:])

        view_margin_x = 2.0 * R
        view_margin_z = max(R, d) * 1.2

        x_grid = np.linspace(-view_margin_x * 1.2, view_margin_x * 1.2, 80)
        z_grid = np.linspace(z_center - view_margin_z, z_center + view_margin_z, 80)
        X, Z = np.meshgrid(x_grid, z_grid)

        Ex = np.zeros_like(X)
        Ez = np.zeros_like(Z)
        ke = 8.987e9

        for i in range(len(points)):
            xi, yi, zi = points[i]
            qi = charges[i]
            rx = X - xi
            rz = Z - zi
            r2 = rx**2 + yi**2 + rz**2 + 1e-4
            r3 = r2**1.5
            Ex += ke * qi * rx / r3
            Ez += ke * qi * rz / r3

        # 2. Отрисовка линий поля
        ax.streamplot(X, Z, Ex, Ez, color='blue', linewidth=1,
                      density=1.8, arrowstyle='->', arrowsize=1.5)

        # 3. Отрисовка электродов
        ax.hlines(z_min, -R, R, colors='red', linewidth=5,
                  label=f'Нижний: {q_bottom:.2e} Кл')
        ax.hlines(z_max, -R, R, colors='darkred', linewidth=5,
                  label=f'Верхний: {q_top:.2e} Кл')


        ax.set_aspect('equal')

        ax.set_xlim(-2.0 * R, 2.0 * R)
        ax.set_ylim(z_center - R, z_center + R)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize='small')
        ax.set_title(f"Силовые линии (разрез XZ, R={R}, d={d})", fontsize=10)