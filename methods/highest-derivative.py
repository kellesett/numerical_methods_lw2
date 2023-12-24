import matplotlib.pyplot as plt
import numpy as np
from equations import tests, tasks


def finite_diff(p, q, f, left, right, certain_cond1, certain_cond2, n=20):
    h = (right - left) / n
    A = np.zeros((n + 1,))
    B = np.zeros((n + 1,))
    C = np.zeros((n + 1,))
    F = np.zeros((n + 1,))
    y = np.zeros((n + 1,))
    aa = np.zeros((n + 1,))
    bb = np.zeros((n + 1,))

    x = np.array([left + h * i for i in range(n + 1)])
    for i in range(n):
        A[i] = 1 / h ** 2 - p(x[i]) / (2 * h)
        C[i] = 1 / h ** 2 + p(x[i]) / (2 * h)
        B[i] = - 2 / h ** 2 + q(x[i])
        F[i] = f(x[i])

    B[0] = certain_cond1[1] * h - certain_cond1[0]
    C[0] = certain_cond1[0]
    F[0] = certain_cond1[2] * h

    B[n] = certain_cond2[1] * h + certain_cond2[0]
    A[n] = -certain_cond2[0]
    F[n] = certain_cond2[2] * h

    aa[0] = -C[0] / B[0]
    bb[0] = F[0] / B[0]
    for i in range(1, n + 1):
        aa[i] = -C[i] / (A[i] * aa[i - 1] + B[i])
        bb[i] = (F[i] - A[i] * bb[i - 1]) / (A[i] * aa[i - 1] + B[i])

    y[n] = (F[n] - bb[n - 1] * A[n]) / (B[n] + aa[n - 1] * A[n])
    for i in range(n, 0, -1):
        y[i - 1] = aa[i - 1] * y[i] + bb[i - 1]
    return x, y

def test(eq: tests.TestDiffEqType2, n: int) -> None:
    fig, ax = plt.subplots()

    x, y = finite_diff(eq.p, eq.q, eq.f, eq.left, eq.right, eq.cc1, eq.cc2, n)
    plt.scatter(x, y, c='red', label='n = {}'.format(n))

    test_x = np.linspace(eq.left, eq.right)
    test_y = [eq.solution(test_x[i]) for i in range(test_x.size)]
    ax.plot(test_x, test_y, c='blue', label='y(x)')
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()


def solve(eq: tasks.DiffEqType2, n: int) -> None:
    fig, ax = plt.subplots()

    x, y = finite_diff(eq.p, eq.q, eq.f, eq.left, eq.right, eq.cc1, eq.cc2, n)
    ax.scatter(x, y, c='red', label='n = {}'.format(n))

    plt.legend(fontsize=12)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    ans = input("Выберите режим работы:\n- (1) тестирование\n- (2) показать решения\n > ")
    n_loc = input("Введите n; (Enter) для значения по умолчанию\n > ")
    if n_loc:
        n_loc = int(n_loc)
    else:
        n_loc = 50
    if ans == '1':
        print("Для завершения программы закройте выплывшее окно")
        test(tests.type2_test2, n_loc)
    elif ans == '2':
        print("Для завершения программы закройте выплывшее окно")
        solve(tasks.type2_eq, n_loc)
    else:
        print("Ошибка ввода")
        exit(0)
