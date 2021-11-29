import time

import numpy as np

import logging

from gps.gmm.gmm import compute_log_obs, logsum
from gps.gmm.gmm_python import GMMPython
from gps.gmm.gmm_sklearn import GMMSklearn

logging.basicConfig(level=logging.DEBUG)

np.random.seed(3)

def step(x, u):
    a = 1.0
    b = 23
    return a * x + b * u + np.random.multivariate_normal(np.zeros(x.shape[0]), np.eye(x.shape[0]) * 0.1)

def gen():
    u_size = len(x0)

    data_u = np.array([np.random.multivariate_normal(np.zeros(u_size), np.eye(u_size) * sigma) for _ in range(ts - 1)])
    data_x = [x0]
    for t in range(ts - 1):
        data_x.append(step(data_x[t], data_u[t]))

    data_x = np.array(data_x)

    return np.c_[data_x[:-1], data_u, data_x[1:]]


ts = 100
sigma = 1.0
x0 = np.zeros(10)

xux = gen()
t = gen()

t1_s = time.time()
gmm1 = GMMPython(max_iterations=100)
gmm1 = gmm1.update(xux, 20)
t1 = time.time() - t1_s

t2_s = time.time()
gmm2 = GMMSklearn(max_iterations=100)
gmm2 = gmm2.update(xux, 20)
t2 = time.time() - t2_s


logobs = compute_log_obs(xux, gmm1.means, gmm1.covariances, gmm1.log_weights)
ll1 = np.sum(logsum(logobs, axis=1))
print("1 scored {} in {}s".format(ll1, t1))

logobs = compute_log_obs(xux, gmm2.means, gmm2.covariances, gmm2.log_weights)
ll2 = np.sum(logsum(logobs, axis=1))
print("2 scored {} in {}s".format(ll2, t2))

print(ll2)

pass