from sklearn.utils.validation import check_random_state
import numpy as np

from cartesian.algorithm import *


def test_algorithm_success(individual):
    cls = type(individual)
    fun = lambda x: 0
    rng = check_random_state(0)
    res = oneplus(fun, random_state=rng, lambda_=4, max_iter=1, f_tol=-1, n_jobs=-1, cls=cls)
    assert res.success == False
    res = oneplus(fun, random_state=rng, lambda_=4, max_iter=0, f_tol=0, n_jobs=-1, cls=cls)
    assert res.success == True
    res = oneplus(fun, random_state=rng, lambda_=4, max_nfev=1, f_tol=-1, n_jobs=-1, cls=cls)
    assert res.success == False


def test_algorithm_twin_problem_with_seed(individual):
    shape = (100, len(individual.inputs))
    x = np.random.normal(size=shape)
    y = individual.fit_transform(x)

    def fun(individual):
        return np.sum((y - individual.fit_transform(x))**2)

    res = oneplus(fun, seed=individual)
    assert res.expr == individual
