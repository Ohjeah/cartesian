from sklearn.utils.validation import check_random_state

from cartesian.algorithm import *


def test_algorithm(individual):
    cls = type(individual)
    fun = lambda x: 0
    rng = check_random_state(0)
    res = oneplus(fun, cls, 1, 1, 1, 1, random_state=rng, lambda_=4, max_iter=1, f_tol=-1, n_jobs=1)
    #assert res.nit = 2
