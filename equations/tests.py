from math import *
from equations import tasks

class TestDiffEqType2(tasks.DiffEqType2):
    def __init__(self, p, q, f, cc1, cc2, solution, left, right):
        super().__init__(p, q, f, cc1, cc2, left, right)
        self.solution = solution

class TestDiffEqType1(tasks.DiffEqType1):
    def __init__(self, f, cc, left, right, solution):
        super().__init__(f, cc, left, right)
        self.solution = solution

class TestDiffEqSystem(tasks.DiffEqSystem):
    def __init__(self, f1, f2, cc, left, right, u, v):
        super().__init__(f1, f2, cc, left, right)
        self.u = u
        self.v = v

type2_test1 = TestDiffEqType2(lambda x: -2,
                              lambda x: -3,
                              lambda x: exp(4 * x),
                              [1, -1, 0.6],
                              [1, 1, 4 * exp(3) + exp(4)],
                              lambda x: exp(-x) + exp(3 * x) + 0.2 * exp(4 * x),
                              0, 1)

type1_test1 = TestDiffEqType1(lambda x, y: (x - x * x) * y,
                              1,0, 1,
                              lambda x: e ** ((-1 / 6) * x * x * (-3 + 2 * x)))

type1_test2 = TestDiffEqType1(lambda x, y: y - y * x,
                              5, 0, 1,
                              lambda x: 5 * exp(-1 / 2 * x * (x - 2)))

system_test1 = TestDiffEqSystem(lambda x, u, v: cos(x + 1.5 * v) - u,
                                lambda x, u, v: -v * v + 2.3 * u - 1.2,
                                [0.25, 1], 0, 3,
                                lambda t: sqrt(t),
                                lambda t: t ** 1/4)