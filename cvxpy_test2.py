# Optimal design over finite number of independent arms
import numpy as np
from cvxpy import *

# fix the random seed for reproducibility
np.random.seed(2)

# number of arms
K = 5

# minimum and maximum variance
var_min = 0.1
var_max = 1
# variances = np.random.uniform(var_min, var_max, size=(K, 1))
variances = np.ones((K, 1))
variances[0] = 0.01
print(np.round(variances, decimals=4))

# allocation over arms
mu = Variable(K)

# objective function
avg_err = matrix_frac(sqrt(variances), diag(mu))

# optimization problem
prob = Problem(Minimize(avg_err),
               [mu >= 0.0,
                sum_entries(mu) == 1.0])
prob.solve()
print("mu =")
print(mu.value)

