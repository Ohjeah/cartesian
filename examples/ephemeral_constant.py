import numpy as np
from sklearn.utils.validation import check_random_state

from cartesian.algorithm import oneplus
from cartesian.cgp import *

rng = check_random_state(1337)
primitives = [
    Primitive("add", np.add, 2),
    Primitive("mul", np.multiply, 2),
    Symbol("x_0"),
    Symbol("x_1"),
    Ephemeral("erc", rng.normal),
]
pset = PrimitiveSet.create(primitives)
x = rng.normal(size=(100, 2))
y = x[:, 1] * x[:, 0] + 0.3
y += 0.05 * rng.normal(size=y.shape)


def func(individual):
    f = compile(individual)
    yhat = f(*x.T)
    return np.sqrt(np.mean((y - yhat) ** 2)) / (y.max() - y.min())


MyCartesian = Cartesian("MyCartesian", pset, n_rows=2, n_columns=3, n_out=1, n_back=1)
res = oneplus(func, cls=MyCartesian, f_tol=0.01, random_state=rng, maxfev=50000, n_jobs=1)
print(res)
