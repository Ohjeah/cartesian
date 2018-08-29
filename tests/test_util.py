import collections

import numpy as np

from cartesian.util import *


def test_make_it():
    x = 1
    assert isinstance(make_it(x), collections.Iterable)
    assert list(make_it(x)) == [x]
    y = [1, 2]
    assert isinstance(make_it(y), collections.Iterable)
    assert list(make_it(y)) == y


def test_replace_nan():
    x = [0, float("nan")]
    np.testing.assert_allclose(replace_nan(x), [0, np.infty])
