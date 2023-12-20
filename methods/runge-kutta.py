import matplotlib.pyplot as plt
import numpy as np
from equations import tests, tasks


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
    return x_lst, y_lst


def system_rk2(eq, length, iterations, x0, y0):
    h = length / iterations
    size = len(eq)
    x_lst, y_lst = np.zeros((iterations,)), np.zeros((2, iterations))
    x, y = x0, y0.copy()

    k1 = np.zeros((size,))
    k2 = np.zeros((size,))
    for i in range(iterations):
        for j in range(size):
            k1[j] = eq[j](x, y[0], y[1])
            k2[j] = eq[j](x + h, y[0] + h * k1[j], y[1] + h * k1[j])
            y[j]= y[j] + (k1[j] + k2[j]) * h / 2
            y_lst[j][i] = y[j]

        x += h
        x_lst[i] = x
    return x_lst, y_lst[0], y_lst[1]


def system_rk4(eq, length, iterations, x0, y0):
    size = len(eq)
    x_lst, y_lst = np.zeros((iterations, )), np.zeros((2, iterations))
    x, y = x0, y0.copy()

    h = length / iterations
    k1 = np.zeros((size,))
    k2 = np.zeros((size,))
    k3 = np.zeros((size,))
    k4 = np.zeros((size,))
    for i in range(iterations):
        for j in range(size):
            k1[j] = eq[j](x, y[0], y[1])
            k2[j] = eq[j](x + h / 2, y[0] + h / 2 * k1[0], y[1] + h / 2 * k1[1])
            k3[j] = eq[j](x + h / 2, y[0] + h / 2 * k2[0], y[1] + h / 2 * k2[1])
            k4[j] = eq[j](x + h, y[0] + h * k3[0], y[1] + h * k3[1])

            y[j] = y[j] + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
            y_lst[j][i] = y[j]

        x += h
        x_lst[i] = x
    return x_lst, y_lst[0], y_lst[1]

def test(eq: tests.TestDiffEqType1, iterations=50):
    fig, ax = plt.subplots()

    test_x = np.linspace(eq.left, eq.right)
    test_y = [eq.solution(test_x[i]) for i in range(test_x.size)]
    ax.plot(test_x, test_y, c='blue', label='y(x)')
    ax.set_xlabel(r'x', fontsize=12, loc="right")
    ax.set_ylabel(r'y(x)', fontsize=12, loc="top", rotation=0)
    x, y = rk2(eq.f, eq.right - eq.left, iterations, eq.left, eq.cc)
    ax.scatter(x, y, c='green', label='RK2', alpha=0.5)
    x, y = rk4(eq.f, eq.right - eq.left, iterations, eq.left, eq.cc)
    ax.scatter(x, y, c='red', label='RK4', alpha=0.5)

    ax.set_title('RK2 & RK4 methods')
    ax.legend()
    ax.grid()
    fig.canvas.manager.set_window_title("Results")
    fig.tight_layout()
    plt.show()


def test_system(eq: tests.TestDiffEqSystem, iterations=30):
    diff_eq = [eq.f1, eq.f2]
    fig, ax = plt.subplots()

    test_t = np.linspace(0, 3)
    test_u = [eq.u(t) for t in test_t]
    test_v = [eq.v(t) for t in test_t]
    ax.plot(test_t, test_u, c='brown', label='u(x)')
    ax.plot(test_t, test_v, c='purple', label='v(x)')
    ax.set_xlabel(r'x', fontsize=12, loc="right")
    ax.set_ylabel(r'y(x)', fontsize=12, loc="top", rotation=0)
    x, y1, y2 = system_rk2(diff_eq, eq.right - eq.left, iterations, eq.left, eq.cc)
    ax.scatter(x, y1, c='green', label='u_RK2')
    ax.scatter(x, y2, c='blue', label='v_RK2')
    x, y1, y2 = system_rk4(diff_eq, eq.right - eq.left, iterations, eq.left, eq.cc)
    ax.scatter(x, y1, c='red', label='u_RK4')
    ax.scatter(x, y2, c='orange', label='v_RK4')

    ax.set_title('RK2 & RK4 methods')
    ax.legend(loc="lower left")
    ax.grid()
    fig.canvas.manager.set_window_title("Results for system")
    fig.tight_layout()
    plt.show()

def solve_system(eq: tasks.DiffEqSystem, iterations=30):
    diff_eq = [eq.f1, eq.f2]
    fig, ax = plt.subplots()

    ax.set_xlabel(r'x', fontsize=12, loc="right")
    ax.set_ylabel(r'y(x)', fontsize=12, loc="top", rotation=0)
    x, y1, y2 = system_rk2(diff_eq, eq.right - eq.left, iterations, eq.left, eq.cc)
    ax.scatter(x, y1, c='green', label='u_RK2', alpha=0.5)
    ax.scatter(x, y2, c='blue', label='v_RK2', alpha=0.5)
    x, y1, y2 = system_rk4(diff_eq, eq.right - eq.left, iterations, eq.left, eq.cc)
    ax.scatter(x, y1, c='red', label='u_RK4', alpha=0.5)
    ax.scatter(x, y2, c='orange', label='v_RK4', alpha=0.5)

    ax.set_title('RK2 & RK4 methods')
    ax.legend(loc="lower left")
    ax.grid()
    fig.canvas.manager.set_window_title("Results for system")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    test(tests.type1_test2, 50)
    solve_system(tasks.system_eq, 1000)
