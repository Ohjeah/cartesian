import numpy as np
from sklearn.utils.validation import check_random_state
from scipy.optimize import minimize, OptimizeResult

from cartesian.algorithm import oneplus
from cartesian.cgp import create_pset, Terminal, Primitive, to_polish, compile, Constant, Base

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
        return np.sqrt(np.mean((y - yhat)**2))/(y.max() - y.min())

    expr, args = to_polish(individual, return_args=True)
    constants = [a for a in args if isinstance(a, Constant)]
    if constants:
        res = minimize(h, np.ones_like(constants))
        individual.consts = res.x
        return res
    else:
        return OptimizeResult(x=(), fun=h(), nfev=1, nit=0, success=True)

Cartesian = type("Cartesian", (Base, ), dict(pset=pset))

success = sum(oneplus(func, Cartesian, 15, 1, 2, 1, f_tol=0.01, random_state=rng, max_nfev=20000, n_jobs=1).success
              for _ in range(30))/30.0

print(success)
