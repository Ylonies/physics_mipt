import numpy as np
import matplotlib.pyplot as plt

# --- параметры системы ---
m = 0.1        # масса шара, кг
k = 1000       # жесткость, Н/м
v0 = 2.0       # начальная скорость к стенке, м/с

# --- расчет ---
omega = np.sqrt(k / m)        # частота
t_c = np.pi / omega           # длительность контакта
t = np.linspace(0, 1.5 * t_c, 500)  # временной массив

# --- аналитическое решение ---
# пока x>0 (контакт), шар сжат:
x = np.where(t <= t_c, (v0 / omega) * np.sin(omega * t), 0)
v = np.where(t <= t_c, v0 * np.cos(omega * t), -v0)  # скорость меняет знак после удара
F = np.where(t <= t_c, -k * x, 0)                    # сила контакта

# --- визуализация ---
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(t, x)
plt.ylabel("Сжатие x(t), м")
plt.grid()

plt.subplot(3,1,2)
plt.plot(t, v)
plt.ylabel("Скорость v(t), м/с")
plt.grid()

plt.subplot(3,1,3)
plt.plot(t, F)
plt.ylabel("Сила F(t), Н")
plt.xlabel("Время t, с")
plt.grid()

plt.suptitle("Модель упругого столкновения шара со стенкой (закон Гука)")
plt.tight_layout()
plt.show()
