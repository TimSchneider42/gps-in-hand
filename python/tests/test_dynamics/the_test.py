from gps.dynamics import DynamicsLR, DynamicsPriorGMM
from gps.sample import SampleList, Sample
from gps.utility.iteration_database import IterationDatabase
import numpy as np
import os
import matplotlib.pyplot as plt

from gps.utility.labeled_data_packer import LabeledDataPacker

run_path = "../output/misc/allegro_pybullet/0541_fpos/data"
run_path = "../output/misc/allegro_pybullet/0643/data"
run_path = "../output/misc/allegro_pybullet/0647/data"
run_path = "../output/misc/allegro_pybullet/0676/data"
# run_path = "../output/misc/box2d_pm_example_lqr/0028/data"
assert os.path.exists(run_path)

db = IterationDatabase(run_path, False, False)
db.load()


def mod_sample(sample):
    fs = {k: v[:, :-1] for k, v in sample.full_state.items()}
    packer = LabeledDataPacker((("pos", 2),))
    return Sample(fs, sample.actions, packer, packer)


def evaluate(min_c_samples, max_clusters, max_samples, strength):
    prior = DynamicsPriorGMM(min_samples_per_cluster=min_c_samples, max_clusters=max_clusters, max_samples=max_samples,
                             strength=strength)
    dynamics = DynamicsLR(prior=prior, regularization=1e-6)

    conditions = [c for c, d in db.reduced_iteration_data[0].cond_data.items() if d.training_trajectory is not None]
    conditions = [0]

    t_sum = 0.0
    for c in conditions:
        print("Condition {}:".format(c))
        c_sum = 0.0
        iterations = db.reduced_iteration_data[:5]
        for itr in iterations:
            samples = [s for s in itr.cond_data[c].training_trajectory.samples]
            test_sample_count = 8
            training_samples = SampleList(samples[:len(samples) - test_sample_count])
            test_samples = SampleList(samples[len(samples) - test_sample_count:])
            dynamics = dynamics.fit(training_samples)

            ts = len(training_samples[0].states) - 1
            diff_sum = np.zeros(ts)
            n_diff_sum = np.zeros(ts)

            for sample in test_samples:
                estimated_states = np.array(
                    [fm @ np.concatenate((x, u)) + fv for x, u, fm, fv in
                     zip(sample.states[:-1], sample.actions, dynamics.Fm, dynamics.fv)])
                actual_states = sample.states[1:]
                diff = np.average(np.abs(estimated_states - actual_states), axis=-1)
                diff_sum += diff

                n_diff = np.average(np.abs(sample.states[:-1] - actual_states), axis=-1)
                n_diff_sum += n_diff

            avg = np.average(diff_sum / ts)
            n_avg = np.average(n_diff_sum / ts)
            print("Iteration {} average: {}\t\t{}".format(itr.iteration_no, avg, n_avg))
            c_sum += avg
        c_avg = c_sum / len(iterations)
        print("Condition {} average: {}".format(c, c_avg))
        t_sum += c_avg
    t_avg = t_sum / len(conditions)
    print("Total average: {}".format(t_avg))
    return t_avg


mu = [1000, 100, 150, 200]
sigma = [200, 0, 0, 0]
titles = ["min_samples_per_cluster", "max_clusters", "max_samples", "strength"]

data = [[] for _ in range(len(mu) + 1)]

figs = [plt.figure() for s in sigma if s > 0]
axs = [f.add_subplot(111) for f in figs]
plt.ion()
plt.show()

a = 1
b = 201
d = b - a
s = 0.5
strengths = [a, b]
for i in range(8):
    strengths += [j for j in np.arange(a + s * d, b, 2 * s * d)]
    s /= 2

val_queue = []

while True:
    if len(val_queue) == 0:
        vals = np.random.normal(mu, sigma)
        while np.any(vals < 0):
            vals = np.random.normal(mu, sigma)
        vals = list(map(int, list(vals[:-1]))) + [vals[-1]]
    else:
        vals = val_queue.pop()
    print("Testing {}".format(vals))
    try:
        for i in range(len(vals)):
            data[i].append(vals[i])
        data[-1].append(evaluate(*vals))

        for i in range(len(vals)):
            if sigma[i] > 0:
                axs[i].clear()
                axs[i].scatter(data[i], data[-1])
                axs[i].set_title(titles[i])
                diff = np.max(data[-1]) - np.min(data[-1])
                axs[i].set_ylim(np.min(data[-1]) - 0.1 * diff, np.max(data[-1]) + 0.1 * diff)
        for i in range(len(vals)):
            if sigma[i] > 0:
                figs[i].canvas.draw()
                figs[i].canvas.flush_events()
    except:
        print("Invalid")
