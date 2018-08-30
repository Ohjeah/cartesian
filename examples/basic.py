import numpy as np
from sklearn.utils.validation import check_random_state

from cartesian.algorithm import oneplus
from cartesian.cgp import *

primitives = [Primitive("add", np.add, 2), Primitive("mul", np.multiply, 2), Symbol("x_0"), Symbol("x_1")]
pset = PrimitiveSet.create(primitives)
rng = check_random_state(42)
x = rng.normal(size=(100, 2))
y = x[:, 1] * x[:, 1] + x[:, 0]
y += 0.05 * rng.normal(size=y.shape)


def func(individual):
    yhat = individual.fit_transform(x)
    return np.sqrt(np.mean((y - yhat) ** 2))


MyCartesian = Cartesian("MyCartesian", pset, n_rows=2, n_columns=3, n_out=1, n_back=1)
res = oneplus(func, f_tol=0.1, cls=MyCartesian, random_state=rng, maxiter=10000, n_jobs=1)
print(res)
print(to_polish(res.expr, return_args=False))
