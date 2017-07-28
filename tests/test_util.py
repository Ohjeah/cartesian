import collections

from cartesian.util import *


def test_make_it():
    x = 1
    assert isinstance(make_it(x), collections.Iterable)
    assert list(make_it(x)) == [x]

    y = [1, 2]
    assert isinstance(make_it(y), collections.Iterable)
    assert list(make_it(y)) == y
