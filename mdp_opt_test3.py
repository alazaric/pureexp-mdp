# Optimal design over finite number of independent states in an MDP
# adding costs (state-dependent)
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

# MDP transition model
p = np.zeros((S, A, S))

# action 0 keeps the state
p[0, 0, 0] = 1.0
p[1, 0, 1] = 1.0

# action 1 triggers transition
p[0, 1, 1] = 1.0
p[1, 1, 0] = 1.0

# convenient matrix representation of the transition model
p_sa = np.reshape(p, (S*A, S))

# cost function is intrinsically defined over the state s', but then it is converted into a more convenient
# state-action function
cost_state = np.zeros(S)
cost_state[0] = 1.0
cost_state[1] = 10.0
cost = np.zeros((S, A))
for s0 in range(S):
    for a in range(A):
        for s1 in range(S):
            cost[s0, a] += cost[s0, a] + p[s0, a, s1]*cost_state[s1]

# weight between error minimization and cost minimization
alpha = 1

# stationary state distribution
mu = Variable(S)
# stationary state-action distribution
rho = Variable(S, A)

# objective function
avg_err_cost = matrix_frac(sqrt(variances), diag(mu)) #+ alpha * vec(cost).T*vec(rho)

# optimization problem
prob = Problem(Minimize(avg_err_cost),
               [mu >= 0.0,
                sum_entries(mu) == 1.0,
                rho >= 0.0,
                sum_entries(rho) == 1.0,
                p_sa.T * vec(rho) == mu,
                rho * np.ones((A, 1)) == mu
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
for s0 in range(S):
    for s1 in range(S):
        for a in range(A):
            P[s0, s1] = P[s0, s1] + p[s0, a, s1] * pi[s0, a]

print('P = ')
print(np.round(P, decimals=5))

# verify that P admits mu as state stationary distribution
np.testing.assert_almost_equal(mu.value.T * P, mu.value.T)

