DEFAULTS = {
    "epsilon_v": 1.2,
    "r_ohm": 2.0,
    "l_h": 5e-3,
    "c_f": 2e-6,
    "u0_v": 1.15,
    "i0_a": 1e-4,
    "t_end_s": 0.03,
    "n_points": 12000,
    "a_a_per_v3": 4e-3,
    "b_a_per_v2": -16e-3,
    "c_a_per_v": 17e-3,
}

LIMITS = {
    "epsilon_v": (0.3, 3.5),
    "r_ohm": (0.1, 100.0),
    "l_h": (1e-5, 0.2),
    "c_f": (1e-8, 1e-3),
    "u0_v": (0.0, 3.5),
    "i0_a": (-1.0, 1.0),
    "t_end_s": (0.001, 1.0),
    "n_points": (1000, 500_000),
}

LIMITS_HELP = """
Допустимые диапазоны (можно вводить краевые значения — расчёт выполнится,
но автогенерация чаще всего при ε≈0.8…2.5 В, R≈0.5…15 Ом, L≈1…50 мГн, C≈0.5…10 мкФ):
  ε, В          0.3 … 3.5       (по умолчанию 1.2)
  R, Ом         0.1 … 100
  L, Гн         1e-5 … 0.2      (по умолчанию 5e-3)
  C, Ф          1e-8 … 1e-3      (по умолчанию 2e-6)
  U(0), В       0 … 3.5         (по умолчанию 1.15)
  I(0), А       -1 … 1
  время, с      0.001 … 1
  число точек   1000 … 500000
Enter — подставить значение по умолчанию.
"""


class InputHandler:
    @staticmethod
    def _fmt_num(x: float | int) -> str:
        if isinstance(x, int):
            return str(x)
        if x == int(x) and abs(x) < 1e4:
            return str(int(x)) if float(x) == int(x) else f"{x:g}"
        return f"{x:g}"

    @staticmethod
    def _read_float(prompt: str, default: float, low: float, high: float) -> float:
        label = (
            f"{prompt} [{InputHandler._fmt_num(default)}], "
            f"диапазон {InputHandler._fmt_num(low)}…{InputHandler._fmt_num(high)}: "
        )
        while True:
            raw = input(label).strip()
            if raw == "":
                return default
            try:
                value = float(raw)
            except ValueError:
                print("Ошибка: введите число. Повторите ввод.")
                continue
            if not (low <= value <= high):
                print(
                    f"Ошибка: нужно от {InputHandler._fmt_num(low)} "
                    f"до {InputHandler._fmt_num(high)}. Повторите ввод."
                )
                continue
            return value

    @staticmethod
    def _read_int(prompt: str, default: int, low: int, high: int) -> int:
        label = f"{prompt} [{default}], диапазон {low}…{high}: "
        while True:
            raw = input(label).strip()
            if raw == "":
                return default
            try:
                value = int(raw)
            except ValueError:
                print("Ошибка: введите целое число. Повторите ввод.")
                continue
            if not (low <= value <= high):
                print(f"Ошибка: нужно от {low} до {high}. Повторите ввод.")
                continue
            return value

    @staticmethod
    def get_parameters():
        print("=== М5. Автогенератор на туннельном диоде ===")
        print(LIMITS_HELP)

        params = {**DEFAULTS}
        params["epsilon_v"] = InputHandler._read_float(
            "ЭДС ε, В", DEFAULTS["epsilon_v"], *LIMITS["epsilon_v"]
        )
        params["r_ohm"] = InputHandler._read_float("R, Ом", DEFAULTS["r_ohm"], *LIMITS["r_ohm"])
        params["l_h"] = InputHandler._read_float("L, Гн", DEFAULTS["l_h"], *LIMITS["l_h"])
        params["c_f"] = InputHandler._read_float("C, Ф", DEFAULTS["c_f"], *LIMITS["c_f"])
        params["u0_v"] = InputHandler._read_float(
            "Начальное U на диоде, В", DEFAULTS["u0_v"], *LIMITS["u0_v"]
        )
        params["i0_a"] = InputHandler._read_float(
            "Начальный I через L, А", DEFAULTS["i0_a"], *LIMITS["i0_a"]
        )
        params["t_end_s"] = InputHandler._read_float(
            "Время моделирования, с", DEFAULTS["t_end_s"], *LIMITS["t_end_s"]
        )
        params["n_points"] = InputHandler._read_int(
            "Число точек", DEFAULTS["n_points"], *LIMITS["n_points"]
        )
        return params
