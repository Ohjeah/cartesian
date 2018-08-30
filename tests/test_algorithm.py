import numpy as np
from sklearn.utils.validation import check_random_state

from cartesian.algorithm import *


def test_algorithm_success(individual):
    cls = type(individual)
    fun = lambda x: 0
    rng = check_random_state(0)
    res = oneplus(fun, random_state=rng, lambda_=4, maxiter=2, f_tol=-1, cls=cls)
    assert not res.success
    res = oneplus(fun, random_state=rng, lambda_=4, maxiter=0, f_tol=0, cls=cls)
    assert res.success
    res = oneplus(fun, random_state=rng, lambda_=4, maxfev=1, f_tol=-1, cls=cls)
    assert not res.success


def test_algorithm_twin_problem_with_seed(individual):
    shape = (100, 2)
    x = np.random.normal(size=shape)
    y = individual.fit_transform(x)

    @optimize_constants
    def fun(f, consts=()):
        return np.sum((y - f(*x.T, *consts)) ** 2)

    res = oneplus(fun, seed=individual)
    assert res.ind == individual


def test_optimize(individual):
    code = [[[3, 1]]]
    outputs = [0]
    ind = type(individual)(code, outputs)
    shape = (100, 2)
    x = np.random.normal(size=shape)

    @optimize_constants
    def fun(f, consts=()):
        return np.sum((f(*x.T, consts)) ** 2)

    res = fun(ind)
    assert res.x < 1e-6
