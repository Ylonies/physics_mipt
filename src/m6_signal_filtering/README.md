# M6: Фильтрация сигналов

Данная программа исследует пассивный **фильтр нижних частот (RC)** и **узкополосный RLC-фильтр** (выходное напряжение на резисторе). Строятся аналитические АЧХ и ФЧХ, численно решаются дифференциальные уравнения цепи (RK4), сравниваются установившиеся режимы при гармоническом и меандровом входе со спектральным методом.

## Физическая модель

### RC (ФНЧ)

Последовательно $R$ и $C$; $U_{\mathrm{in}}$ на цепи, $U_{\mathrm{out}}$ на конденсаторе. На низких частотах $X_C = 1/(\omega C)$ велико — сигнал проходит; на высоких конденсатор шунтирует выход.

### RLC (полосовой, ★)

Последовательный контур $R$–$L$–$C$; $U_{\mathrm{out}} = RI$. На резонансе $\omega_0 = 1/\sqrt{LC}$ ток максимален.

## Вывод аналитических формул

### RC: от уравнения цепи к $H(\omega)$

По 2-му правилу Кирхгофа:

$$i = C\,\frac{dU_{\mathrm{out}}}{dt}, \qquad U_{\mathrm{in}} = Ri + U_{\mathrm{out}}.$$

Подставляя $i$:

$$RC\,\frac{dU_{\mathrm{out}}}{dt} + U_{\mathrm{out}} = U_{\mathrm{in}}(t).$$

Для $U_{\mathrm{in}} = U_0 e^{j\omega t}$, $U_{\mathrm{out}} = U_0 H(\omega) e^{j\omega t}$:

$$H(\omega) = \frac{1}{1 + j\omega RC}.$$

**АЧХ и ФЧХ:**

$$|H(\omega)| = \frac{1}{\sqrt{1 + (\omega RC)^2}}, \qquad \varphi(\omega) = -\arctan(\omega RC).$$

**Частота среза** ($-3$ дБ): $\omega_c = 1/(RC)$.

**Переходная характеристика** (скачок $U_{\mathrm{in}} = U_0\,\Theta(t)$, $U_{\mathrm{out}}(0)=0$):

$$h(t) = U_0\bigl(1 - e^{-t/(RC)}\bigr), \quad t \ge 0.$$

**Установившийся отклик** на $U_{\mathrm{in}}(t) = U_0\sin(\omega t)$:

$$U_{\mathrm{out}}(t) = U_0|H(\omega)|\sin\bigl(\omega t + \varphi(\omega)\bigr).$$

### RLC: передаточная функция на резисторе

$U_{\mathrm{in}} = RI + L\dot I + U_C$, $\dot U_C = I/C$, $U_{\mathrm{out}} = RI$. Импеданс $Z = R + j\omega L + 1/(j\omega C)$, ток $I = U_0/Z$:

$$H(\omega) = \frac{R}{Z(\omega)} = \frac{j\omega RC}{1 - \omega^2 LC + j\omega RC}.$$

**Резонанс:** $\omega_0 = 1/\sqrt{LC}$, $|H(\omega_0)| = 1$, $\varphi(\omega_0) = 0$.

**Добротность:** $Q = \dfrac{1}{R}\sqrt{\dfrac{L}{C}}$.

### Меандр: спектральный метод

Симметричный меандр $\pm U_0$, период $T$, $\omega_0 = 2\pi/T$:

$$U_{\mathrm{in}}(t) = \frac{4U_0}{\pi}\sum_{k=1,3,5,\ldots}^{\infty} \frac{\sin(k\omega_0 t)}{k}.$$

$$U_{\mathrm{out}}(t) = \sum_{k=1,3,5,\ldots} \frac{4U_0}{\pi k}\,|H(k\omega_0)|\,\sin\bigl(k\omega_0 t + \varphi(k\omega_0)\bigr).$$

## Входные параметры

Параметры в `config.py`:

1. **RC:** $R$, $C$, $U_0$, частоты $\omega/\omega_c$, период меандра $T$.
2. **RLC (★):** $R$, $L$, $C$, $U_0$, $\omega/\omega_0$.
3. **Численный метод:** RK4, $\Delta t \ll \min(\tau,\, T/N)$.

По умолчанию: $R=1$ кОм, $C=1$ мкФ $\Rightarrow$ $\tau=1$ мс, $f_c\approx159$ Гц; $L=10$ мГн, $C=1$ мкФ $\Rightarrow$ $f_0\approx1.59$ кГц, $Q=10$.

## Запуск модели

Сначала в консоль — таблицы сравнения с теорией; затем выбор графиков:

```bash
uv run python -m src.m6_signal_filtering.main
```

```
  a — RC (фильтр нижних частот)
  b — RLC (полосовой)
  ab — оба набора (отдельные окна)
  n — без графиков
```

Сохранение PNG без GUI:

```bash
MPLBACKEND=Agg uv run python -m src.m6_signal_filtering.main --save-plots output/m6_plots
```

## Результаты расчёта

* **RC:** АЧХ/ФЧХ, переходный процесс (гармоника), свип $A$ и $\varphi$, меандр и спектр.
* **RLC (★):** АЧХ/ФЧХ у резонанса, гармоники, меандр с $T \approx 2\pi/\omega_0$.
* Каждый фильтр — **отдельный полноэкранный набор** (основной + меандр).

## Структура кода

| Файл / каталог | Назначение |
|----------------|------------|
| `config.py` | Параметры по умолчанию |
| `main.py` | Точка входа |
| `signal_filter.py` | Консоль → выбор a/b/ab/n |
| `models/transfer.py` | $H(\omega)$ |
| `models/integrator.py` | RK4 |
| `models/signals.py` | Гармоника, меандр |
| `models/spectrum.py` | $A$, $\varphi$, спектр |
| `models/simulation.py` | Прогоны |
| `models/result_visualizer.py` | Графики RC / RLC |

## Тесты

```bash
PYTHONPATH=. uv run pytest tests/m6_signal_filtering/analyt_tests.py -v
```

## Авторы

Торокулова Жанылай, Б13-402 — функции расчёта и вычисления, тесты \
Яренкова Софья, Б13-402 — визуализация
