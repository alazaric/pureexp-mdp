import numpy as np
from cvxopt import solvers, matrix, spdiag, log


def acent(A, b):
    m, n = A.size

    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (n,1))
        if min(x) <= 0.0: return None
        f = -sum(log(x))
        Df = -(x**-1).T
        if z is None: return f, Df
        H = spdiag(z[0] * x**-2)
        return f, Df, H

    return solvers.cp(F, A=A, b=b)['x']


m = 5
Apy = np.diag(np.ones(5))
A = matrix(Apy, (5,5))
print(A)

bpy = np.ones(m)
b = matrix(bpy, (5,1))
print(b)

print(acent(A, b))