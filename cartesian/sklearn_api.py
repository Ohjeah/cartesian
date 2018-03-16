import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_array, check_random_state

from .cgp import create_pset, Symbol, Primitive, Constant, compile, Cartesian
from .algorithm import oneplus, optimize

DEFAULT_PRIMITIVES = [
    Primitive("add", np.add, 2),
    Primitive("mul", np.multiply, 2)
]


def _ensure_1d(yhat, shape):
    try:
        yhat.shape[1]
        return yhat
    except:
        return np.ones(shape) * yhat


class evaluate:  # ugly construct s.th. you can pickle it and use joblib
    def __init__(self, x, y, metric):
        self.n_samples, *out = y.shape
        self.multi_output = True if out else False
        self.x = x
        self.y = y
        self.metric = metric

    def error(self, f, consts=()):
        if self.multi_output:
            yhat = np.array([
                _ensure_1d(i, self.n_samples) for i in f(*self.x.T, *consts)
            ]).T
        else:
            yhat = _ensure_1d(f(*self.x.T, *consts), self.n_samples)
        return self.metric(self.y, yhat)

    def __call__(self, individual):
        return optimize(self.error, individual)


class Symbolic(BaseEstimator, RegressorMixin):
    """Wraps the 1 + lambda algorithm in sklearn api"""

    def __init__(self,
                 operators=None,
                 n_const=0,
                 n_rows=1,
                 n_columns=3,
                 n_back=1,
                 max_iter=1000,
                 max_nfev=10000,
                 lambda_=4,
                 f_tol=0,
                 seed=None,
                 random_state=None,
                 n_jobs=1,
                 metric=mean_squared_error):
        """
        :param operators: list of primitive excluding terminals
        :param n_const: number of symbolic constants
        :param n_rows: number of rows in the code block
        :param n_columns: number of columns in the code block
        :param n_back: number of rows to look back for connections
        :param metric: what to optimize for
        :param fun: `callable(individual)`, function to be optimized
        :param random_state: an instance of np.random.RandomState, a seed integer or None
        :param cls: The base class for individuals
        :type cls: (optional) instance of cartesian.cgp.Cartesian
        :param seed: (optional) can be passed instead of cls.
        :param lambda_: number of offspring per generation
        :param max_iter: maximum number of generations
        :param max_nfev: maximum number of function evaluations. Important, if fun is another optimizer
        :param f_tol: threshold for precision
        :param n_jobs: number of jobs for joblib embarrassingly easy parallel
        """
        self.operators = DEFAULT_PRIMITIVES or operators
        self.constants = [Constant("c_{}".format(i)) for i in range(n_const)]
        self.n_rows = n_rows
        self.n_back = n_back
        self.n_columns = n_columns
        self.n_out = None
        self.pset = None
        self.res = None
        self.model = None

        # parameters for algorithm
        self.max_nfev = max_nfev
        self.max_iter = max_iter
        self.lambda_ = lambda_
        self.f_tol = f_tol
        self.metric = metric
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(self, x, y):
        x = check_array(x)
        _, self.n_out = y.reshape(y.shape[0], -1).shape
        _, n_features = x.shape
        terminals = [Symbol("x_{}".format(i)) for i in range(n_features)]
        self.pset = create_pset(self.operators + terminals + self.constants)
        cls = Cartesian(
            str(hash(self)),
            self.pset,
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            n_out=self.n_out,
            n_back=self.n_back)

        self.res = oneplus(
            evaluate(x, y, self.metric),
            random_state=self.random_state,
            cls=cls,
            lambda_=self.lambda_,
            max_iter=self.max_iter,
            max_nfev=self.max_nfev,
            f_tol=self.f_tol,
            n_jobs=self.n_jobs,
            seed=self.seed)

        self.model = compile(self.res.expr)
        return self

    def predict(self, x):
        if self.n_out > 1:
            yhat = np.array([
                _ensure_1d(i, x.shape[0])
                for i in self.model(*x.T, *self.res.x)
            ]).T
        else:
            yhat = _ensure_1d(self.model(*x.T, *self.res.x), x.shape[0])
        return yhat
