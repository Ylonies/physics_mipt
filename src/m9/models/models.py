"""Трассировка лучей: подзорная труба (тонкие и толстые сферические линзы)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Ray2D:
    x: float
    y: float
    ux: float
    uy: float
    weight: float
    n: float


@dataclass
class ThinLens:
    position: float
    focal_length: float
    aperture: float


@dataclass
class ThickLens:
    """Симметричная биконвексная линза: две сферические поверхности."""
    center_x: float
    radius_curvature: float
    thickness: float
    refractive_index: float
    aperture: float


class PhysicsModels:
    @staticmethod
    def thin_lens_image_distance(s: float, f: float) -> float:
        if abs(s - f) < 1e-12:
            return math.inf
        return 1.0 / (1.0 / f - 1.0 / s)

    @staticmethod
    def thin_lens_magnification(s: float, sp: float) -> float:
        return -sp / s

    @staticmethod
    def radius_from_lensmaker(f: float, thickness: float, n: float) -> float:
        """Радиус симметричной биконвексной линзы из f и толщины (линзовое уравнение)."""
        if f <= 0:
            raise ValueError("Фокусное расстояние должно быть положительным.")
        a = (n - 1.0) * thickness / n
        b = 2.0 * (n - 1.0)
        c = -1.0 / f
        # (n-1)t/(n R²) + 2(n-1)/R − 1/f = 0  →  c R² + b R + a = 0,  c = −1/f
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return 2.0 * f
        r = (-b - math.sqrt(disc)) / (2.0 * c)
        return r if r > 0 else (-b + math.sqrt(disc)) / (2.0 * c)

    @staticmethod
    def _normalize(ux: float, uy: float) -> tuple[float, float]:
        norm = math.hypot(ux, uy)
        if norm < 1e-15:
            return 1.0, 0.0
        return ux / norm, uy / norm

    @staticmethod
    def _refract_direction(ux: float, uy: float, nx: float, ny: float, n1: float, n2: float) -> tuple[float, float] | None:
        cos_i = -(ux * nx + uy * ny)
        if cos_i < 0:
            nx, ny = -nx, -ny
            cos_i = -cos_i
        sin2_i = max(0.0, 1.0 - cos_i * cos_i)
        eta = n1 / n2
        sin2_t = eta * eta * sin2_i
        if sin2_t > 1.0:
            return None
        cos_t = math.sqrt(1.0 - sin2_t)
        rx = eta * ux + (eta * cos_i - cos_t) * nx
        ry = eta * uy + (eta * cos_i - cos_t) * ny
        return PhysicsModels._normalize(rx, ry)

    @staticmethod
    def _brightness_factor(n1: float, cos1: float, n2: float, cos2: float) -> float:
        return (n1 * abs(cos1)) / (n2 * abs(cos2) + 1e-15)

    @classmethod
    def propagate_to_x(cls, ray: Ray2D, x_target: float) -> Ray2D | None:
        if abs(ray.ux) < 1e-12:
            return None
        t = (x_target - ray.x) / ray.ux
        if t < -1e-12:
            return None
        return Ray2D(
            x=ray.x + t * ray.ux,
            y=ray.y + t * ray.uy,
            ux=ray.ux,
            uy=ray.uy,
            weight=ray.weight,
            n=ray.n,
        )

    @classmethod
    def intersect_sphere(cls, ray: Ray2D, vertex_x: float, radius: float, n_out: float) -> Ray2D | None:
        """Сферическая поверхность: вершина vertex_x, центр в (vertex_x + R, 0)."""
        cx = vertex_x + radius
        ox, oy = ray.x - cx, ray.y
        dx, dy = ray.ux, ray.uy
        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - radius * radius
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        t_candidates = [t for t in (t1, t2) if t > 1e-10]
        if not t_candidates:
            return None
        t_hit = min(t_candidates)

        px = ray.x + t_hit * dx
        py = ray.y + t_hit * dy
        nx = (px - cx) / radius
        ny = py / radius
        n1, n2 = ray.n, n_out
        dirs = cls._refract_direction(dx, dy, nx, ny, n1, n2)
        if dirs is None:
            return None
        ux2, uy2 = dirs
        cos1 = abs(dx * nx + dy * ny)
        cos2 = abs(ux2 * nx + uy2 * ny)
        w2 = ray.weight * cls._brightness_factor(n1, cos1, n2, cos2)
        return Ray2D(px, py, ux2, uy2, w2, n2)

    @classmethod
    def pass_thin_lens_paraxial(cls, ray: Ray2D, lens: ThinLens) -> Ray2D | None:
        if abs(ray.y) > lens.aperture:
            return None
        theta = math.atan2(ray.uy, ray.ux)
        theta_new = theta - ray.y / lens.focal_length
        ux, uy = cls._normalize(math.cos(theta_new), math.sin(theta_new))
        cos_old = max(abs(math.cos(theta)), 1e-9)
        cos_new = max(abs(math.cos(theta_new)), 1e-9)
        w_new = ray.weight * cos_old / cos_new
        return Ray2D(ray.x, ray.y, ux, uy, w_new, ray.n)

    @classmethod
    def pass_thick_lens(cls, ray: Ray2D, lens: ThickLens) -> Ray2D | None:
        if abs(ray.y) > lens.aperture:
            return None
        vertex1 = lens.center_x - lens.thickness / 2.0
        vertex2 = lens.center_x + lens.thickness / 2.0
        r1 = lens.radius_curvature
        r2 = -lens.radius_curvature

        after_s1 = cls.intersect_sphere(ray, vertex1, r1, lens.refractive_index)
        if after_s1 is None:
            return None
        after_s2 = cls.intersect_sphere(after_s1, vertex2, r2, 1.0)
        return after_s2

    @staticmethod
    def build_spotting_scope_thin(params: dict) -> tuple[list[ThinLens], float, dict]:
        lens1 = ThinLens(params["lens1_pos_m"], params["f_obj_m"], params["aperture_obj_m"])
        lens2 = ThinLens(params["lens2_pos_m"], params["f_eye_m"], params["aperture_eye_m"])
        s1 = params["object_distance_m"]
        sp1 = PhysicsModels.thin_lens_image_distance(s1, lens1.focal_length)
        m1 = PhysicsModels.thin_lens_magnification(s1, sp1)
        intermediate = lens1.position + sp1
        s2 = lens2.position - intermediate
        sp2 = PhysicsModels.thin_lens_image_distance(s2, lens2.focal_length)
        m2 = PhysicsModels.thin_lens_magnification(s2, sp2)
        optics = {
            "sp1": sp1,
            "sp2": sp2,
            "s2": s2,
            "m1": m1,
            "m2": m2,
            "m_total": m1 * m2,
            "intermediate_x": intermediate,
            "image_plane": lens2.position + sp2,
        }
        return [lens1, lens2], optics["image_plane"], optics

    @staticmethod
    def build_spotting_scope_thick(params: dict) -> tuple[list[ThickLens], float, dict]:
        n = params["lens_index"]
        t = params["lens_thickness_m"]
        r_obj = PhysicsModels.radius_from_lensmaker(params["f_obj_m"], t, n)
        r_eye = PhysicsModels.radius_from_lensmaker(params["f_eye_m"], t, n)
        lens1 = ThickLens(params["lens1_pos_m"], r_obj, t, n, params["aperture_obj_m"])
        lens2 = ThickLens(params["lens2_pos_m"], r_eye, t, n, params["aperture_eye_m"])
        _, image_plane, optics = PhysicsModels.build_spotting_scope_thin(params)
        return [lens1, lens2], image_plane, optics

    @classmethod
    def trace_thin(cls, ray: Ray2D, lenses: list[ThinLens], image_plane: float) -> tuple[Ray2D | None, list[tuple[float, float]]]:
        path = [(ray.x, ray.y)]
        current = ray
        for lens in lenses:
            dist = lens.position - current.x
            if dist < -1e-12:
                return None, path
            if dist > 1e-12:
                current = cls.propagate_to_x(current, current.x + dist)
                if current is None:
                    return None, path
                path.append((current.x, current.y))
            current = cls.pass_thin_lens_paraxial(current, lens)
            if current is None:
                return None, path
            path.append((current.x, current.y))
        current = cls.propagate_to_x(current, image_plane)
        if current is None:
            return None, path
        path.append((current.x, current.y))
        return current, path

    @classmethod
    def trace_thick(cls, ray: Ray2D, lenses: list[ThickLens], image_plane: float) -> tuple[Ray2D | None, list[tuple[float, float]]]:
        path = [(ray.x, ray.y)]
        current = ray
        for lens in lenses:
            dist = (lens.center_x - lens.thickness / 2.0) - current.x
            if dist < -1e-12:
                return None, path
            if dist > 1e-12:
                current = cls.propagate_to_x(current, current.x + dist)
                if current is None:
                    return None, path
                path.append((current.x, current.y))
            current = cls.pass_thick_lens(current, lens)
            if current is None:
                return None, path
            path.append((current.x, current.y))
            gap = (lens.center_x + lens.thickness / 2.0) - current.x
            if gap > 1e-12:
                current = cls.propagate_to_x(current, current.x + gap)
                if current is None:
                    return None, path
                path.append((current.x, current.y))
        current = cls.propagate_to_x(current, image_plane)
        if current is None:
            return None, path
        path.append((current.x, current.y))
        return current, path

    def _accumulate_image(
        self,
        params: dict,
        lenses_thin: list[ThinLens] | None,
        lenses_thick: list[ThickLens] | None,
        image_plane: float,
        m_theory: float,
        use_thin: bool,
    ) -> dict:
        half_angle = math.radians(params["object_half_angle_deg"])
        angles = np.linspace(-half_angle, half_angle, params["rays_per_point"])
        y_obj = np.linspace(-params["object_height_m"], params["object_height_m"], params["grid_n"])
        n_bins = params["image_bins"]
        y_max = params["object_height_m"] * abs(m_theory) * 1.5 + 1e-6

        image = np.zeros((n_bins, n_bins))
        paths: list[list[tuple[float, float]]] = []
        dtheta = (2 * half_angle) / max(params["rays_per_point"] - 1, 1)

        trace_fn = self.trace_thin if use_thin else self.trace_thick
        lenses = lenses_thin if use_thin else lenses_thick

        for y0 in y_obj:
            for theta0 in angles:
                w0 = dtheta * max(math.cos(theta0), 0.05)
                ux, uy = self._normalize(math.cos(theta0), math.sin(theta0))
                ray = Ray2D(0.0, y0, ux, uy, w0, 1.0)
                final, path = trace_fn(ray, lenses, image_plane)
                if final is None:
                    continue
                if len(paths) < 50:
                    paths.append(path)
                iy = int((final.y + y_max) / (2 * y_max) * (n_bins - 1))
                if 0 <= iy < n_bins:
                    image[iy, n_bins // 2] += final.weight

        if image.max() > 0:
            image /= image.max()

        chief_lo = trace_fn(Ray2D(0.0, y_obj[0], 1.0, 0.0, 1.0, 1.0), lenses, image_plane)[0]
        chief_hi = trace_fn(Ray2D(0.0, y_obj[-1], 1.0, 0.0, 1.0, 1.0), lenses, image_plane)[0]
        m_ray = m_theory
        if chief_lo and chief_hi:
            m_ray = (chief_hi.y - chief_lo.y) / (y_obj[-1] - y_obj[0])

        return {"image": image, "ray_paths": paths, "m_ray": m_ray}

    def _simulate(self, params: dict) -> dict:
        thin_lenses, image_plane, optics = self.build_spotting_scope_thin(params)
        m_theory = optics["m_total"]
        use_thin = params["lens_mode"] == "1"

        base = {
            "image_plane": image_plane,
            "optics": optics,
            "sp_theory": optics["sp2"],
            "m_theory": m_theory,
            "thin_lenses": thin_lenses,
            "geometry_ok": params["lens2_pos_m"] > optics["intermediate_x"],
        }

        if use_thin:
            data = self._accumulate_image(params, thin_lenses, None, image_plane, m_theory, use_thin=True)
            return {**base, "lens_mode": "1", "image": data["image"], "ray_paths": data["ray_paths"], "m_ray": data["m_ray"]}

        thick_lenses, _, _ = self.build_spotting_scope_thick(params)
        data = self._accumulate_image(params, None, thick_lenses, image_plane, m_theory, use_thin=False)
        return {**base, "lens_mode": "2", "image": data["image"], "ray_paths": data["ray_paths"], "m_ray": data["m_ray"]}

    def get_model(self, params):
        name = "Тонкие линзы" if params["lens_mode"] == "1" else "Толстые линзы"
        return self._simulate, name
