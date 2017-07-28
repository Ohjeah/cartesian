import operator

import pytest

from cartesian.cgp import *


@pytest.fixture()
def pset():
    terminals = [Terminal("x_0"), Terminal("x_1")]
    operators = [Primitive("neg", operator.neg, 1)]
    pset = create_pset(terminals + operators)
    return pset


@pytest.fixture(params=[pset])
def individual(request):
    pset = request.param()
    MyCartesian = type("MyCartesian", (Base, ), {"pset": pset})
    code = [[[2, 0]]]
    outputs = [2]
    return MyCartesian(code, outputs, 1)
