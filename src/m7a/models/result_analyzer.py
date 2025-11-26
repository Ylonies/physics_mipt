import numpy as np
from .config import SimulationConfig
class ResultAnalyzer:
    @staticmethod
    def compute_density_profile(pos, nbins=50):
        cfg = SimulationConfig
        hist, edges = np.histogram(pos[:,0], bins=nbins, range=(0, cfg.Lx))
        bin_w = edges[1] - edges[0]
        density = hist / (bin_w * cfg.Ly)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, density