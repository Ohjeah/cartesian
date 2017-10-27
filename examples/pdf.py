from functools import partial
import inspect

import numpy as np
from sklearn.utils.validation import check_random_state

from cartesian.algorithm import oneplus
from cartesian.cgp import *

rng = check_random_state(0)

def random():
    caller = inspect.stack()[1]
    l = len(caller.frame.f_locals["x_0"])
    return rng.normal(size=(100, l))

primitives = [
    Primitive("add", np.add, 2),
    Primitive("mul", np.multiply, 2),
    Symbol("x_0"),
    Symbol("x_1"),
    Primitive("r", random, 0),
]

pset = create_pset(primitives)

x = rng.normal(size=(100, 2))
y = x[:, 0] * x[:, 1] + 0.3
y += 0.05 * rng.normal(size=y.shape)


def func(individual):
    f = compile(individual)
    yhat = f(*x.T)
    return np.sqrt(np.mean((y - yhat)**2))/(y.max() - y.min())


MyCartesian = Cartesian("MyCartesian", pset, n_rows=3, n_columns=4, n_out=1, n_back=1)
ind = MyCartesian.create(rng)
print(to_polish(ind))

f = compile(ind)

import matplotlib.pyplot as plt
yhat = f(*x.T)
plt.errorbar(x.T[0], yhat.mean(axis=0), yerr=yhat.std(axis=0))
plt.show()
