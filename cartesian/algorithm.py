import math
from functools import wraps

import numpy as np
from sklearn.utils.validation import check_random_state
from scipy.optimize import OptimizeResult, minimize

from joblib import Parallel, delayed

from .cgp import active_gene_mutation, compile, to_polish, Constant


def return_opt_result(f, individual):
    """Ensure that f returns a scipy.optimize.OptimizeResults

    Args:
        f: callable(individual)
        individual:  instance of cartesian.cgp.Base

    Returns:
        OptimizeResult

    """
    res = f(individual)
    if not isinstance(res, OptimizeResult):
        res = OptimizeResult(x=(), fun=res, nit=0, nfev=1, success=True)
    return res


def oneplus(
    fun, random_state=None, cls=None, lambda_=4, maxiter=100, maxfev=None, f_tol=0, n_jobs=1, seed=None, callback=None
):
    """1 + lambda algorithm.

    In each generation, create lambda offspring and compare their fitness to the parent individual.
    The fittest individual carries over to the next generation. In case of a draw, the offspring is prefered.

    Args:
        fun: callable(individual), function to be optimized
        random_state: an instance of np.random.RandomState, a seed integer or None
        cls: base class for individuals
        lambda_: number of offspring per generation
        maxiter: maximum number of generations
        maxfev: maximum number of function evaluations. Important, if fun is another optimizer
        f_tol: absolute error in metric(ind) between iterations that is acceptable for convergence
        n_jobs: number of jobs for joblib embarrassingly easy parallel
        seed: (optional) can be passed instead of cls, used for hot-starts
        callback: callable(OptimizeResult), can be optionally used to monitor progress

    Returns:
        scipy.optimize.OptimizeResult with non-standard attributes res.x = values for constants res.expr = expression res.fun = best value for the function

    """
    maxiter = maxfev if maxfev else maxiter
    maxfev = maxfev or math.inf
    random_state = check_random_state(random_state)
    best = seed or cls.create(random_state=random_state)
    best_res = return_opt_result(fun, best)
    nfev = best_res.nfev
    res = OptimizeResult(ind=best, x=best_res.x, fun=best_res.fun, nit=0, nfev=nfev, success=False, expr=str(best))
    if best_res.fun <= f_tol:
        res["success"] = True
        return res

    for i in range(1, maxiter):
        offspring = [active_gene_mutation(best, random_state=random_state) for _ in range(lambda_)]
        with Parallel(n_jobs=n_jobs) as parallel:
                offspring_fitness = parallel(delayed(return_opt_result)(fun, o) for o in offspring)
        # offspring_fitness = [return_opt_result(fun, o) for o in offspring]
        best, best_res = min(zip(offspring + [best], offspring_fitness + [best_res]), key=lambda x: x[1].fun)
        nfev += sum(of.nfev for of in offspring_fitness)
        res = OptimizeResult(ind=best, x=best_res.x, fun=best_res.fun, nit=i, nfev=nfev, success=False, expr=str(best))
        if callback is not None:
            callback(res)
        if res.fun <= f_tol:
            res["success"] = True
            return res

        elif res.nfev >= maxfev:
            return res

    return res


def optimize(fun, individual):
    """Prepares individual and fun to optimize fun(c | individual)

    Args:
        fun: callable of lambda expression and its constant values.
        individual:

    Returns:
        scipy.optimize.OptimizeResult

    """
    f = compile(individual)

    def h(consts=()):
        return fun(f, consts)

    expr, args = to_polish(individual, return_args=True)
    constants = [a for a in args if isinstance(a, Constant)]
    if constants:
        res = minimize(h, np.ones_like(constants))
        individual.consts = res.x
        return res
    else:
        return OptimizeResult(x=(), fun=h(), nfev=1, nit=0, success=True)


def optimize_constants(fun):
    """Wrap a measure with constant optimization."""

    @wraps(fun)
    def inner(individual):
        res = optimize(fun, individual)
        return res

    return inner
