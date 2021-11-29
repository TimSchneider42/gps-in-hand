import numpy as np
import os

from sklearn.linear_model import LinearRegression

from gps.algorithm.dynamics import DynamicsLR, DynamicsPriorGMM
from gps.sample.sample_collator import SampleCollator
from gps.sample.sample_list import SampleList
from gps.utility.labeled_data_packer import LabeledDataPacker

np.set_printoptions(precision=3, suppress=True)

fx = np.array([[-0.9, 0], [-0.5, -0.5]])
fu = np.array([[3, 0], [0, 1]])
fc = np.array([0.2, 0])
x0 = np.array([3, 2])
dx = x0.shape[0]
du = fu.shape[1]
p = False
pc = 20
tc = 5

ts = 100
packer = LabeledDataPacker([("dummy", dx)])

# Generate samples
samples = []
for j in range(pc + tc):
    sc = SampleCollator(packer, du, ts)
    current_state = x0 + np.random.normal(size=(dx,))
    sc.set_initial_state({"dummy": current_state})
    if p:
        print(f"x0: {x0}")
    for i in range(ts - 1):
        u = np.random.normal(size=(du,))
        noise = np.random.normal(size=(dx,))
        current_state = fx @ current_state + fu @ u + fc
        if p:
            print(f"Step {i}:")
            print(f"u: {u}")
            print(f"n: {noise}")
            print(f"x: {current_state}")
        sc.add(u, {"dummy": current_state})
    samples.append(sc.finalize())

sl_p = SampleList(samples[:pc])
sl = SampleList(samples[pc:])
dynamics = DynamicsLR().fit(sl)
print(f"Fm: {os.linesep}{dynamics.Fm[0]}")
print(f"Fv: {os.linesep}{dynamics.fv[0]}")
print()

dynp = DynamicsLR(prior=DynamicsPriorGMM()).fit(sl_p)
dyn = dynp.fit(sl)
print(f"Fm: {os.linesep}{dyn.Fm[0]}")
print(f"Fv: {os.linesep}{dyn.fv[0]}")

xu = np.concatenate((sl.states[:, 0, :], sl.actions[:, 0, :]), axis=-1).reshape((-1, dx + du))
x = sl.states[:, [1], :].reshape((-1, dx))

r = LinearRegression()
r.fit(xu, x)

print()
print(r.coef_)
print(r.intercept_)