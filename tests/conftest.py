import operator

import pytest

from cartesian.cgp import *


@pytest.fixture()
def pset():
    terminals = [Terminal("x_0"), Terminal("x_1"), Constant("c")]
    operators = [Primitive("neg", operator.neg, 1)]
    pset = create_pset(terminals + operators)
    return pset


@pytest.fixture(params=[pset])
def individual(request):
    pset = request.param()
    MyCartesian = Cartesian("MyCartesian", pset, n_columns=1)
    code = [[[3, 1]]]
    outputs = [3]
    return MyCartesian(code, outputs)

@pytest.fixture
def sc():
    s = Structual("SC", (lambda x, y: x/y), 2)
    return s
