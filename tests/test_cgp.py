import operator

from cgp.cgp import *


def test_PrimitiveSet():
    terminals = [Terminal("x_0")]
    operators = []

    pset = create_pset(terminals + operators)

    assert pset.terminals = terminals
    assert pset.operators = operators
    assert pset.mapping = {1: "x_0"}
    assert pset.max_arity == 0
