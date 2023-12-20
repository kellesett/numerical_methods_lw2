from math import *

class DiffEquation:
    def __init__(self, p, q, f):
        self.p = p
        self.q = q
        self.f = f

eq1 =  DiffEquation(lambda x: -2,
                    lambda x: -3,
                    lambda x: exp(4 * x))