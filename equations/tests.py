from math import *

class TestDiffEquation:
    def __init__(self, p, q, f, solution):
        self.p = p
        self.q = q
        self.f = f

        self.solution = solution


test1 = TestDiffEquation(lambda x: -2,
                          lambda x: -3,
                          lambda x: exp(4 * x),
                          lambda x: exp(-x) + exp(3 * x) + 0.2 * exp(4 * x))