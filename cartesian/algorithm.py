import math
from operator import itemgetter
from functools import wraps

from sklearn.utils.validation import check_random_state
from scipy.optimize import OptimizeResult
from joblib import Parallel, delayed

from .cgp import Base, point_mutation

def return_opt_result(f):
    @wraps(f)
    def inner(individual):
        fitness = f(individual)
        return OptimizeResult(x=(), fun=fitness, nit=0, nfev=1, success=True)
    return inner


def return_opt_result(f, individual):
    res = f(individual)
    if not isinstance(res, OptimizeResult):
        res = OptimizeResult(x=(), fun=res, nit=0, nfev=1, success=True)
    return res



def oneplus(fun, cls, n_columns, n_rows, n_back, n_out, random_state=None,
            lambda_=4, max_iter=100, max_nfev=None, f_tol=0, n_jobs=1):

    max_iter = max_nfev if max_nfev else max_iter
    max_nfev = max_nfev or math.inf

    random_state = check_random_state(random_state)

    best = cls.create(n_columns, n_rows, n_back, n_out, random_state=random_state)
    best_res = return_opt_result(fun, best)

    nfev = best_res.nfev
    res = OptimizeResult(expr=best, x=best_res.x, fun=best_res.fun, nit=0, nfev=nfev, success=False)

    if best_res.fun <= f_tol:
        res["success"] = False
        return res

    for i in range(1, max_iter):
        offspring = [point_mutation(best, random_state=random_state) for _ in range(lambda_)]

        with Parallel(n_jobs=n_jobs) as parallel:
                offspring_fitness = parallel(delayed(return_opt_result)(fun, o) for o in offspring)

        best, best_res = min(zip(offspring + [best], offspring_fitness + [best_res]), key=lambda x: x[1].fun)
        nfev += sum(of.nfev for of in offspring_fitness)

        res = OptimizeResult(expr=best, x=best_res.x, fun=best_res.fun, nit=i, nfev=nfev, success=False)
        if res.fun <= f_tol:
            res["success"] = True
            return res
        elif res.nfev >= max_nfev:
            return res

    return res
