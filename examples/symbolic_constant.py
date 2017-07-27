import numpy as np
from sklearn.utils.validation import check_random_state
from scipy.optimize import minimize

from cgp.algorithm import oneplus
from cgp.cgp import create_pset, Terminal, Primitive, to_polish, compile, Constant

primitives = [
    Primitive("add", np.add, 2),
    Primitive("mul", np.multiply, 2),
    Terminal("x_0"),
    Terminal("x_1"),
    Constant("c"),
]

pset = create_pset(primitives)
rng = check_random_state(None)

x = rng.normal(size=(100, 2))
y = x[:, 1] * x[:, 0] + 0.3
#y += 0.05 * rng.normal(size=y.shape)

def func(individual):
    f = compile(individual)
    def h(*consts):
        yhat = f(*x.T, *consts)
        return np.sqrt(np.mean((y - yhat)**2))

    expr, args = to_polish(individual, return_args=True)
    constants = [a for a in args if isinstance(a, Constant)]
    if constants:
        res = minimize(h, np.ones_like(constants))
        individual.consts = res.x
        return res.fun or np.infty
    else:
        return h()


res, fitness = oneplus(func, pset, 2, 2, 2, 1, f_tol=0.1, random_state=rng, max_iter=1000)
print(res, fitness)
print(to_polish(res, return_args=False))
