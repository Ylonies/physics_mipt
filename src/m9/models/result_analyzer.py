import numpy as np


class ResultAnalyzer:
    @staticmethod
    def analyze(solution, params, model_name):
        m_theory = float(solution["m_theory"])
        m_ray = float(solution["m_ray"])
        rel_err = abs(m_ray - m_theory) / (abs(m_theory) + 1e-12)

        image = solution["image"]
        n_bins = image.shape[0]
        y_axis = np.linspace(-1, 1, n_bins)
        profile = image[:, n_bins // 2]
        if profile.max() > 0:
            profile = profile / profile.max()

        return {
            "model_name": model_name,
            "lens_mode": params["lens_mode"],
            "optics": solution["optics"],
            "sp_theory": float(solution["sp_theory"]),
            "m_theory": m_theory,
            "m_ray": m_ray,
            "rel_err_magnification": float(rel_err),
            "y_axis": y_axis,
            "image": image,
            "profile": profile,
            "ray_paths": solution["ray_paths"],
            "image_plane": solution["image_plane"],
            "thin_lenses": solution["thin_lenses"],
            "params": params,
        }
