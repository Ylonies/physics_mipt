import numpy as np

class GridGenerator:
    # Класс для создания расчетной сетки электродов

    def generate_single_disk(self, R, z_coord, n_rings):
        points = []
        areas = []
        dr = R / n_rings

        points.append([0.0, 0.0, z_coord])
        areas.append(np.pi * (dr / 2)**2)

        # 2. Кольцевые слои
        for k in range(1, n_rings):
            r_in = (k - 0.5) * dr
            r_out = (k + 0.5) * dr
            if k == n_rings - 1: r_out = R

            r_k = (r_in + r_out) / 2

            n_sectors = int(2 * np.pi * r_k / dr)
            if n_sectors < 4: n_sectors = 4
            d_phi = 2 * np.pi / n_sectors

            s_sector = np.pi * (r_out**2 - r_in**2) / n_sectors

            for j in range(n_sectors):
                phi = j * d_phi
                points.append([r_k * np.cos(phi), r_k * np.sin(phi), z_coord])
                areas.append(s_sector)

        return np.array(points), np.array(areas)

    def create_system(self, R, d, V, n_rings):
        # Генерирует полную систему из двух дисков с потенциалами
        pts_up, areas_up = self.generate_single_disk(R, d/2, n_rings)
        pots_up = np.full(len(pts_up), V / 2)

        pts_down, areas_down = self.generate_single_disk(R, -d/2, n_rings)
        pots_down = np.full(len(pts_down), -V / 2)

        all_points = np.vstack([pts_up, pts_down])
        all_areas = np.concatenate([areas_up, areas_down])
        all_potentials = np.concatenate([pots_up, pots_down])

        return all_points, all_areas, all_potentials