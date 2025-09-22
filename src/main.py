import math
import numpy as np
from scipy.integrate import solve_ivp

g = 9.81

#обработка входных значений 

v0 = float(input("Начальная скорость (м/с): "))
if not (0 < v0 <= 200):
    raise ValueError("Начальная скорость должна быть в диапазоне (0, 200] м/с.")

alpha_deg = float(input("Угол броска в градусах: "))
if not (0 < alpha_deg < 90):
    raise ValueError("Угол броска должен быть в пределах (0, 90] градусов.")

gamma = float(input("Коэффициент сопротивления: "))
if gamma < 0:
    raise ValueError("Коэффициент сопротивления не может быть отрицательным.")

m = float(input("Введите массу тела (кг): "))
if m <= 0:
    raise ValueError("Масса должна быть положительной.")


alpha = math.radians(alpha_deg) 

v0_x = v0 * math.cos(alpha)
v0_y = v0 * math.sin(alpha)

t_span = (0.1, 10)  
x0 = [0, v0_x]
y0 = [0, v0_y]
t_eval = np.linspace(t_span[0], t_span[1], 10)


# первая часть
def x_t(t, X):
    x, vx = X
    dx_dt = vx
    dvx_dt = - (gamma / m) * vx
    return [dx_dt, dvx_dt]

def y_t(t, Y):
    y, vy = Y
    dy_dt = vy
    dvy_dt = - g - (gamma / m) * vy  
    return [dy_dt, dvy_dt]


# примерное решение

sol_x = solve_ivp(x_t, t_span, x0, t_eval=t_eval)
sol_y = solve_ivp(y_t, t_span, y0, t_eval=t_eval)
print(sol_x.y[0])
print(sol_y.y[0])


# вторая часть

def x_t_2(t, X):
    x, vx = X
    dx_dt = vx 
    dvx_dt = - (gamma / m) * abs(vx) * vx
    return [dx_dt, dvx_dt]

def y_t_2(t, Y):
    y, vy = Y
    dy_dt = vy
    dvy_dt = - g - (gamma / m) * abs(vy) * vy
    return [dy_dt, dvy_dt]


sol_x_2 = solve_ivp(x_t_2, t_span, x0, t_eval=t_eval)
sol_y_2 = solve_ivp(y_t_2, t_span, y0, t_eval=t_eval)
print(sol_x_2.y[0])
print(sol_y_2.y[0])