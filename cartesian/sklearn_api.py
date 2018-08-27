import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_array, check_random_state

from .cgp import PrimitiveSet, Symbol, Primitive, Constant, compile, Cartesian
from .algorithm import oneplus, optimize

DEFAULT_PRIMITIVES = [Primitive("add", np.add, 2), Primitive("mul", np.multiply, 2)]


def _ensure_1d(yhat, shape):
    try:
        yhat.shape[1]
        return yhat

    except (AttributeError, TypeError, IndexError):
        return np.ones(shape) * yhat


class _Evaluate:  # ugly construct s.th. you can pickle it and use joblib
    def __init__(self, x, y, metric):
        """Wraps metric for optimization"""
        self.n_samples, *out = y.shape
        self.multi_output = True if out else False
        self.x = x
        self.y = y
        self.metric = metric

    def error(self, f, consts=()):
        if self.multi_output:
            yhat = np.array([_ensure_1d(i, self.n_samples) for i in f(*self.x.T, *consts)]).T
        else:
            yhat = _ensure_1d(f(*self.x.T, *consts), self.n_samples)
        return self.metric(self.y, yhat)

    def __call__(self, individual):
        return optimize(self.error, individual)


class Symbolic(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        operators=None,
        n_const=0,
        n_rows=1,
        n_columns=3,
        n_back=1,
        max_iter=1000,
        max_nfev=10000,
        lambda_=4,
        f_tol=0,
        seeded_individual=None,
        random_state=None,
        n_jobs=1,
        metric=None,
    ):
        """Wraps the 1 + lambda algorithm in sklearn api.

        Note:
            n_costs provides a convenience method to create Symbols.
            All constants can be directly passed via the operators.

        Args:
            operators: list of primitives
            n_const: number of symbolic constants
            n_rows: number of rows in the code block
            n_columns: number of columns in the code block
            n_back: number of rows to look back for connections
            max_iter: maximum number of generations
            max_nfev: maximum number of function evaluations. Important, if fun is another optimizer
            lambda_: number of offspring per generation
            f_tol: Absolute error in metric(ind) between iterations that is acceptable for convergence
            seed: an individual used to hot-start the optimization
            random_state: an instance of np.random.RandomState, an integer used as seed, or None
            n_jobs: number of jobs for joblib embarrassingly easy parallel
            metric: callable(individual), function to be optimized
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
        self.metric = metric if metric is not None else mean_squared_error
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs
        self.seeded_individual = seeded_individual

    def fit(self, x, y):
        """Trains the model given the regression task.

        Args:
            x (np.ndarray): input data matrix for fitting of size (number_of_input_points, number_of_features)
            y (np.ndarray): target data vector for fitting of size (number_of_input_points)

        Returns: self

        """
        x = check_array(x)
        _, self.n_out = y.reshape(y.shape[0], -1).shape
        _, n_features = x.shape
        terminals = [Symbol("x_{}".format(i)) for i in range(n_features)]
        self.pset = PrimitiveSet.create(self.operators + terminals + self.constants)
        cls = Cartesian(
            str(hash(self)),
            self.pset,
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            n_out=self.n_out,
            n_back=self.n_back,
        )
        self.res = oneplus(
            _Evaluate(x, y, self.metric),
            random_state=self.random_state,
            cls=cls,
            lambda_=self.lambda_,
            max_iter=self.max_iter,
            max_nfev=self.max_nfev,
            f_tol=self.f_tol,
            n_jobs=self.n_jobs,
            seed=self.seeded_individual,
        )
        self.model = compile(self.res.expr)
        return self

    def predict(self, x):
        """Use the fitted model f to make a prediction.

        Args:
            x: input data matrix for scoring

        Returns: predicted target data vector

        """
        if self.n_out > 1:
            yhat = np.array([_ensure_1d(i, x.shape[0]) for i in self.model(*x.T, *self.res.x)]).T
        else:
            yhat = _ensure_1d(self.model(*x.T, *self.res.x), x.shape[0])
        return yhat
