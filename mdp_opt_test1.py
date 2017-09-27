# Optimal design over finite number of independent states in an MDP
import numpy as np
from cvxpy import *

# fix the random seed for reproducibility
np.random.seed(2)

# number of states
S = 2
# number of actions
A = 2

# state noise variance
variances = np.ones((S, 1))
variances[0] = 1
variances[1] = 4.0

print("variances = ")
print(np.round(variances, decimals=5))

# we first compute the optimal allocation without the "MDP constraint"
# optimal allocation
mu_unconstrained = Variable(S)

# objective function
avg_err_unconstrained = matrix_frac(sqrt(variances), diag(mu_unconstrained))

# optimization problem
prob = Problem(Minimize(avg_err_unconstrained),
               [mu_unconstrained >= 0.0,
                sum_entries(mu_unconstrained) == 1.0,
                ])
prob.solve()

print("mu_unconstrained =")
print(mu_unconstrained.value)

# MDP transition model: deterministic from s0 to s1 whatever action is taken
# probability of reaching a (fixed) state given the starting state and the action
p_s0 = np.zeros((S, A))
p_s0[0, 0] = 1.0
p_s0[1, 1] = 1.0

p_s1 = np.zeros((S, A))
p_s1[0, 1] = 1.0
p_s1[1, 0] = 1.0

# stationary state distribution
mu = Variable(S)
# stationary state-action distribution
rho = Variable(S, A)

# objective function
avg_err = matrix_frac(sqrt(variances), diag(mu))

# optimization problem
prob = Problem(Minimize(avg_err),
               [mu >= 0.0,
                sum_entries(mu) == 1.0,
                rho >= 0.0,
                sum_entries(rho) == 1.0,
                vec(rho).T * vec(p_s0) == mu[0],
                vec(rho).T * vec(p_s1) == mu[1],
                rho[0, :] * np.ones((A, 1)) == mu[0],
                rho[1, :] * np.ones((A, 1)) == mu[1]
                ])
prob.solve()

print("mu =")
print(mu.value)
print("rho =")
print(rho.value)

# recover the policy corresponding to the stationary state-action distribution
pi = np.zeros((S,  A))
for s in range(S):
    pi[s, :] = rho.value[s, :]/mu.value[s]

print("pi =")
print(pi)

np.testing.assert_almost_equal(sum(pi[0, :]), 1.0)
np.testing.assert_almost_equal(sum(pi[1, :]), 1.0)

# compute the MC associated to the policy
P = np.zeros((S, S))
for s in range(S):
    P[s, 0] = p_s0[s, 0] * pi[s, 0] + p_s0[s, 1] * pi[s, 1]
    P[s, 1] = p_s1[s, 0] * pi[s, 0] + p_s1[s, 1] * pi[s, 1]

print('P = ')
print(np.round(P, decimals=5))

# verify that P admits mu as state stationary distribution
np.testing.assert_almost_equal(mu.value.T * P, mu.value.T)

