from math import *

class DiffEqType2:
    def __init__(self, p, q, f, cc1, cc2, left, right):
        self.p = p
        self.q = q
        self.f = f
        self.cc1 = cc1
        self.cc2 = cc2
        self.left = left
        self.right = right

class DiffEqType1:
    def __init__(self, f, cc, left, right):
        self.f = f
        self.cc = cc
        self.left = left
        self.right = right

class DiffEqSystem:
    def __init__(self, f1, f2, cc, left, right):
        self.f1 = f1
        self.f2 = f2
        self.left = left
        self.right = right
        self.cc = cc


type2_eq =  DiffEqType2(lambda x: -1,
                        lambda x: 2 / x,
                        lambda x: x + 0.4,
                        [-0.5, 1, 2],
                        [1, 0, 4],
                        1.1, 1.4)

system_eq = DiffEqSystem(lambda x, u, v: exp(-(u ** 2 + v ** 2)) + 2 * x,
                         lambda x, u, v: 2 * u ** 2 + v,
                         [0.5, 1], 0, 3)

system_eq1 = DiffEqSystem(lambda x, u, v: cos(x + 1.5 * v) - u,
                                lambda x, u, v: -v * v + 2.3 * u - 1.2,
                                [0.25, 1], 0, 3)