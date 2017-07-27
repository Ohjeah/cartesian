from operator import itemgetter

from sklearn.utils.validation import check_random_state

from .cgp import Base, point_mutation


def oneplus(fun, pset, n_columns, n_rows, n_back, n_out, random_state=None, lambda_=4, max_iter=100, f_tol=0):
    random_state = check_random_state(random_state)
    Cartesian = type("Cartesian", (Base, ), dict(pset=pset))

    best = Cartesian.create(n_columns, n_rows, n_back, n_out, random_state=random_state)
    fitness = fun(best)

    if fitness <= f_tol:
        return best, fitness

    for i in range(max_iter):
        offspring = [point_mutation(best, random_state=random_state) for _ in range(lambda_)]
        offspring_fitness = [fun(o) for o in offspring]
        best, fitness = min(zip(offspring + [best], offspring_fitness + [fitness]), key=itemgetter(1))
        if fitness <= f_tol:
            return best, fitness

    return best, fitness
