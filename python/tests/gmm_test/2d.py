import time

import numpy as np

import logging

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.pyplot import scatter, figure, title

from gps.gmm.gmm import compute_log_obs, logsum
from gps.gmm.gmm_python import GMMPython
from gps.gmm.gmm_sklearn import GMMSklearn

logging.basicConfig(level=logging.DEBUG)


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


np.random.seed(3)

num_clusters = 2
num_datapoints = 50 * num_clusters

mu = np.random.uniform(0, 10, (num_clusters, 2))
sigma = np.random.uniform(0, 1, (num_clusters, 2, 2))
for s in sigma:
    while any(np.linalg.eigh(s)[0] <= 0):
        s += np.diag([0.1] * 2)
weights = np.random.uniform(0, 1, num_clusters)
weights /= sum(weights)

data = {}

for c in range(num_clusters):
    data[c] = np.array([np.random.multivariate_normal(mu[c], sigma[c]) for _ in
                        range(min(int(num_datapoints * weights[c]),
                                  num_datapoints - sum([len(d) for d in data.values()])))])

data_arr = np.concatenate(list(data.values()))

t1_s = time.time()
gmm1 = GMMPython(max_iterations=100)
gmm1 = gmm1.update(data_arr, num_clusters)
t1 = time.time() - t1_s

t2_s = time.time()
gmm2 = GMMSklearn(max_iterations=100)
gmm2 = gmm2.update(data_arr, num_clusters)
t2 = time.time() - t2_s

logobs = compute_log_obs(data_arr, gmm1.means, gmm1.covariances, gmm1.log_weights)
ll1 = np.sum(logsum(logobs, axis=1))
print("1 scored {} in {}s".format(ll1, t1))

logobs = compute_log_obs(data_arr, gmm2.means, gmm2.covariances, gmm2.log_weights)
ll2 = np.sum(logsum(logobs, axis=1))
print("2 scored {} in {}s".format(ll2, t2))

print(ll2)

color = {c: np.random.rand(3) for c in range(num_clusters)}

title("original")
for c, d in data.items():
    p = scatter(d[:, 0], d[:, 1], zorder=2, color=color[c])
    plot_cov_ellipse(sigma[c], mu[c], zorder=1, alpha=0.25, color=color[c])

figure()
title("python")
for c, d in data.items():
    p = scatter(d[:, 0], d[:, 1], zorder=2, color=color[c])
    plot_cov_ellipse(gmm1.covariances[c], gmm1.means[c], zorder=1, alpha=0.25, color="grey")

figure()
title("sklearn")
for c, d in data.items():
    p = scatter(d[:, 0], d[:, 1], zorder=2, color=color[c])
    plot_cov_ellipse(gmm2.covariances[c], gmm2.means[c], zorder=1, alpha=0.25, color="grey")

print(f"Original weights: {weights}")
print(f"Python weights:   {gmm1.weights.reshape((-1,))}")
print(f"Sklearn weights:  {gmm2.weights}")

gmm1.inference(data_arr)

plt.show()
