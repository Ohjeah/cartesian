import operator

import pytest

from cartesian.cgp import *


@pytest.fixture()
def pset():
    terminals = [Symbol("x_0"), Symbol("x_1"), Constant("c")]
    operators = [Primitive("neg", operator.neg, 1)]
    pset = PrimitiveSet.create(terminals + operators)
    return pset


@pytest.fixture(params=[pset])
def individual(request):
    pset = request.param()
    MyCartesian = Cartesian("MyCartesian", pset, n_columns=1)
    code = [[[3, 1]]]
    outputs = [3]
    return MyCartesian(code, outputs)  # y = -x0


@pytest.fixture
def sc():
    s = Structural("SC", (lambda x, y: x / y), 2)
    return s
