# Generate data for worst-case risk analysis.
import numpy as np


np.random.seed(2)
n = 5
mu = np.abs(np.random.randn(n, 1))/15
Sigma = np.random.uniform(-.15, .8, size=(n, n))
Sigma_nom = Sigma.T.dot(Sigma)
print("Sigma_nom =")
print(np.round(Sigma_nom, decimals=2))

# Form and solve portfolio optimization problem.
# Here we minimize risk while requiring a 0.1 return.
from cvxpy import *
w = Variable(n)
w_sum = Variable(1)
ret = mu.T*w
risk = quad_form(w, Sigma_nom)
prob = Problem(Minimize(risk),
               [w_sum == 1,
                w_sum == sum_entries(w),
                w >= 0,
                ret >= 0.1,
                norm(w, 1) <= 2])
prob.solve()
print("w =")
print(np.round(w.value, decimals=2))