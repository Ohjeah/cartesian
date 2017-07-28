import operator
import pickle

import numpy as np
import pytest

from cartesian.cgp import *


def test_PrimitiveSet(pset):
    assert pset.mapping == {0: pset.terminals[0], 1: pset.terminals[1], 2: pset.operators[0]}
    assert pset.max_arity == 1
    assert pset.context[pset.operators[0].name] == operator.neg


def test_Cartesian(individual):
    x = np.ones((1, 2))
    y = individual.fit_transform(x)
    assert y == np.array([-1])


def test_Cartesian_get(individual):
    assert individual[0] == 0
    assert individual[1] == 1


def test_Cartesian_set(individual):
    n = len(individual)
    individual[n-1] = 1
    assert individual.outputs[0] == 1


def test_to_polish(individual):
    polish, used_arguments = to_polish(individual)
    assert polish == ["neg(x_0)"]
    assert len(used_arguments) == 1


def test_boilerplate(individual):
    assert boilerplate(individual) == "lambda x_0, x_1:"
    assert boilerplate(individual, used_arguments=[individual.pset.terminals[0]]) == "lambda x_0:"


def test_compile(individual):
    f = compile(individual)
    assert f(1, 1) == -1


def test_point_mutation(individual):
    new_individual = point_mutation(individual)

    assert new_individual.inputs is not individual.inputs
    assert new_individual.inputs == individual.inputs
    assert new_individual.code is not individual.code
    assert new_individual.outputs is not individual.outputs

    changes = 0
    if new_individual.outputs != individual.outputs:
        changes += 1
    for c1, c2 in zip(individual.code, new_individual.code):
        for c11, c22 in zip(c1, c2):
            if c11 != c22:
                changes += 1

    assert 0 <= changes <= 1

@pytest.mark.xfail
def test_Cartesian_pickle(individual):
    assert individual == pickle.loads(pickle.dumps(individual))
