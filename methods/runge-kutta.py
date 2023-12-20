import matplotlib.pyplot as plt
import numpy as np
from math import *

def f1(x, y):
    return (x - x * x) * y


def g1(x, u, v):
    return cos(x + 1.5 * v) - u


def g2(x, u, v):
    return -v * v + 2.3 * u - 1.2


def rk2(f, length, iterations, x0, y0):
    x_lst, y_lst = list(), list()
    x, y = x0, y0

    x_lst.append(x)
    y_lst.append(y)

    h = length / iterations
    for i in range(iterations):
        y = y + (f(x, y) + f(x + h, y + h * f(x, y))) * h / 2
        x += h
        x_lst.append(x)
        y_lst.append(y)

    # plt.scatter(x_lst, y_lst, c='green', label='RK2')
    return x_lst, y_lst


def rk4(f, length, iterations: int, x0: float, y0: float):
    x_lst, y_lst = list(), list()
    x, y = x0, y0

    x_lst.append(x)
    y_lst.append(y)

    h = length / iterations
    for i in range(iterations):
        k1 = f(x, y)
        k2 = f(x + h / 2, y + h / 2 * k1)
        k3 = f(x + h / 2, y + h / 2 * k2)
        k4 = f(x + h, y + h * k3)

        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h
        x_lst.append(x)
        y_lst.append(y)

    # plt.scatter(x_lst, y_lst, c='red', label='RK4')
    return x_lst, y_lst


def system_rk2(equations, length, iterations, x0, y0):
    h = length / iterations
    size = len(equations)
    x_lst, y_lst = np.zeros((iterations,)), np.zeros((2, iterations))
    x, y = x0, y0.copy()

    k1 = np.zeros((size,))
    k2 = np.zeros((size,))
    for i in range(iterations):
        for j in range(size):
            k1[j] = equations[j](x, y[0], y[1])
            k2[j] = equations[j](x + h, y[0] + h * k1[j], y[1] + h * k1[j])
            y[j]= y[j] + (k1[j] + k2[j]) * h / 2
            y_lst[j][i] = y[j]

        x += h
        x_lst[i] = x
    return x_lst, y_lst[0], y_lst[1]


def system_rk4(equations, length, iterations, x0, y0):
    size = len(equations)
    x_lst, y_lst = np.zeros((iterations, )), np.zeros((2, iterations))
    x, y = x0, y0.copy()

    h = length / iterations
    k1 = np.zeros((size,))
    k2 = np.zeros((size,))
    k3 = np.zeros((size,))
    k4 = np.zeros((size,))
    for i in range(iterations):
        for j in range(size):
            k1[j] = equations[j](x, y[0], y[1])
            k2[j] = equations[j](x + h / 2, y[0] + h / 2 * k1[0], y[1] + h / 2 * k1[1])
            k3[j] = equations[j](x + h / 2, y[0] + h / 2 * k2[0], y[1] + h / 2 * k2[1])
            k4[j] = equations[j](x + h, y[0] + h * k3[0], y[1] + h * k3[1])

            y[j] = y[j] + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
            y_lst[j][i] = y[j]

        x += h
        x_lst[i] = x
    return x_lst, y_lst[0], y_lst[1]

def test():
    iterations = 50
    fig, ax = plt.subplots()

    test_x = np.linspace(0, 1)
    test_y = e ** ((-1 / 6) * test_x * test_x * (-3 + 2 * test_x))
    ax.plot(test_x, test_y, c='blue', label='y(x)')
    ax.set_xlabel(r'x', fontsize=12, loc="right")
    ax.set_ylabel(r'y(x)', fontsize=12, loc="top", rotation=0)
    x, y = rk2(f1, 1, iterations, 0, 1)
    ax.scatter(x, y, c='green', label='RK2', alpha=0.5)
    x, y = rk4(f1, 1, iterations // 2, 0, 1)
    ax.scatter(x, y, c='red', label='RK4', alpha=0.5)

    ax.set_title('RK2 & RK4 methods')
    ax.legend()
    ax.grid()
    fig.canvas.manager.set_window_title("Results")
    fig.tight_layout()
    plt.show()

def test_system(f1, f2, t0, u0, v0, u, v, iterations=30):
    diff_equation = [f1, f2]
    certain_cond = [u0, v0]

    fig, ax = plt.subplots()

    test_t = np.linspace(0, 3)
    test_u = [u(t) for t in test_t]
    test_v = [v(t) for t in test_t]
    ax.plot(test_t, test_u, c='brown', label='u(x)')
    ax.plot(test_t, test_v, c='purple', label='v(x)')
    ax.set_xlabel(r'x', fontsize=12, loc="right")
    ax.set_ylabel(r'y(x)', fontsize=12, loc="top", rotation=0)
    x, y1, y2 = system_rk2(diff_equation, 3, iterations, t0, certain_cond)
    ax.scatter(x, y1, c='green', label='u_RK2')
    ax.scatter(x, y2, c='blue', label='v_RK2')
    x, y1, y2 = system_rk4(diff_equation, 3, iterations, t0, certain_cond)
    ax.scatter(x, y1, c='red', label='u_RK4')
    ax.scatter(x, y2, c='orange', label='v_RK4')

    ax.set_title('RK2 & RK4 methods')
    ax.legend(loc="lower left")
    ax.grid()
    fig.canvas.manager.set_window_title("Results for system")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # test()
    test_system(g1, g2, 0, 0.25, 1, lambda t: sqrt(t), lambda t: t ** 1/4, 30)
