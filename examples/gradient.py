from functools import partial
import autograd.numpy as np
from autograd import grad
from sklearn.utils import check_random_state
from scipy.optimize import minimize

from cartesian.algorithm import *
from cartesian.cgp import *

primitives = [
    Primitive("add", np.add, 2),
    Primitive("mul", np.multiply, 2),
    Symbol("x_0"),
    Symbol("x_1"),
    Constant("c_1"),
    Constant("c_2"),
]

pset = create_pset(primitives)
rng = check_random_state(1337)
rng = np.random

x = rng.normal(size=(100, 2))
y = x[:, 1] * x[:, 0] + 0.3
y += 0.05 * rng.normal(size=y.shape)
y = np.array(y)
def optimize(fun, individual):
    f = compile(individual)
    def h(consts=()):
        return fun(f, consts)

    expr, args = to_polish(individual, return_args=True)
    constants = [a for a in args if isinstance(a, Constant)]
    if constants:
        res = minimize(value_and_grad(h), np.ones_like(constants), jac=True)
        individual.consts = res.x
        return res
    else:
        return OptimizeResult(x=(), fun=h(), nfev=1, nit=0, success=True)


def loss(f, consts):
    f = partial(f, *x.T)
    yhat = f(*consts)
    return np.mean((y - yhat)**2)
    #return np.sqrt(np.mean((y - yhat)**2))/(y.max() - y.min())



MyCartesian = Cartesian("MyCartesian", pset, n_rows=2, n_columns=3, n_out=1, n_back=1)

ind = MyCartesian.create(random_state=rng)

@grad
def h(consts):
    return loss(compile(ind), consts)

print(h(np.ones(1)))
#print(optimize(loss, individual))
