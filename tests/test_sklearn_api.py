import pytest

from sklearn.datasets import make_regression

from cartesian import Symbolic

@pytest.mark.parametrize("n_out", [1, 2])
def test_Symbolic_fit(n_out):
    x, y = make_regression(n_features=2, n_informative=1, n_targets=n_out)
    est = Symbolic(max_nfev=1, lambda_=1).fit(x, y)
    yhat = est.predict(x)
    assert yhat.shape == y.shape


def test_Symbolic_joblib():
    x, y = make_regression(n_features=2, n_informative=1, n_targets=1)
    yhat = Symbolic(n_jobs=-1, max_nfev=1, lambda_=1).fit(x, y).predict(x)
    assert yhat.shape == y.shape
