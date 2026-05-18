DEFAULTS = {
    "object_distance_m": 0.25,
    "object_height_m": 0.01,
    "lens_thickness_m": 0.012,
    "f_obj_m": 0.2,
    "f_eye_m": 0.05,
    "lens2_pos_m": 1.31,
    "lens_index": 1.5,
    "object_half_angle_deg": 5.0,
    "rays_per_point": 19,
    "grid_n": 50,
    "image_bins": 150,
}


class InputHandler:
    @staticmethod
    def _read_float(prompt: str, default: float, low: float, high: float) -> float:
        while True:
            raw = input(f"{prompt} [{default}]: ").strip()
            if raw == "":
                return default
            try:
                value = float(raw)
            except ValueError:
                print("Ошибка: введите число. Повторите ввод.")
                continue
            if not (low <= value <= high):
                print(f"Ошибка: значение должно быть в диапазоне [{low} .. {high}]. Повторите ввод.")
                continue
            return value

    @staticmethod
    def _read_lens_mode() -> str:
        while True:
            lens_mode = input("Режим [1]: ").strip() or "1"
            if lens_mode in ("1", "2"):
                return lens_mode
            print("Ошибка: выберите 1 или 2. Повторите ввод.")

    @staticmethod
    def get_parameters():
        print("=== М9. Подзорная труба ===")
        print("1 — тонкие линзы   2 — толстые сферические линзы")
        lens_mode = InputHandler._read_lens_mode()
        print("Enter — значение по умолчанию в скобках.\n")

        s = InputHandler._read_float(
            "Расстояние от объекта до объектива, м",
            DEFAULTS["object_distance_m"],
            0.05,
            2.0,
        )
        f_obj = InputHandler._read_float(
            "Фокусное расстояние объектива, м",
            DEFAULTS["f_obj_m"],
            0.02,
            1.0,
        )
        f_eye = InputHandler._read_float(
            "Фокусное расстояние окуляра, м",
            DEFAULTS["f_eye_m"],
            0.01,
            0.3,
        )
        lens2_pos = InputHandler._read_float(
            "Положение окуляра по оси, м",
            DEFAULTS["lens2_pos_m"],
            s + 0.05,
            3.0,
        )
        h = InputHandler._read_float(
            "Полувысота объекта, м",
            DEFAULTS["object_height_m"],
            1e-3,
            0.05,
        )

        params = {**DEFAULTS, "lens_mode": lens_mode}
        params["object_distance_m"] = s
        params["f_obj_m"] = f_obj
        params["f_eye_m"] = f_eye
        params["lens1_pos_m"] = s
        params["lens2_pos_m"] = lens2_pos
        params["object_height_m"] = h
        params["aperture_obj_m"] = max(0.02, 4.0 * h)
        params["aperture_eye_m"] = max(0.012, 3.0 * h)

        if lens_mode == "2":
            t_mm = InputHandler._read_float(
                "Толщина линзы, мм",
                DEFAULTS["lens_thickness_m"] * 1000,
                1.0,
                40.0,
            )
            params["lens_thickness_m"] = t_mm / 1000.0

        return params
