import operator
import pickle

import numpy as np
import pytest
import hypothesis
from hypothesis.strategies import integers

from cartesian.cgp import *
from cartesian.cgp import _get_valid_inputs


def test_PrimitiveSet(pset):
    assert pset.mapping == {
        1: pset.terminals[0],
        2: pset.terminals[1],
        0: pset.terminals[2],
        3: pset.operators[0]
    }
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
    individual[n - 1] = 1
    assert individual.outputs[0] == 1


def test_to_polish(individual):
    polish, used_arguments = to_polish(individual)
    assert polish == ["neg(x_0)"]
    assert len(used_arguments) == 1


def test_boilerplate(individual):
    assert boilerplate(individual) == "lambda x_0, x_1, c:"
    assert boilerplate(
        individual,
        used_arguments=[individual.pset.terminals[0]]) == "lambda x_0:"


def test_compile(individual):
    f = compile(individual)
    assert f(1, 1) == -1


def test_point_mutation(individual):
    for _ in range(20):
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


def test_Cartesian_pickle(individual):
    pickled = pickle.loads(pickle.dumps(individual))
    for k in individual.__dict__.keys():
        assert pickled.__dict__[k] == individual.__dict__[k]


def test_Cartesian_copy(individual):
    individual.memory[0] = 1
    new = individual.clone()
    with pytest.raises(KeyError):
        new.memory[0]

    assert new.code == individual.code
    assert new.code is not individual.code
    assert new.outputs == individual.outputs
    assert new.outputs is not individual.outputs


def test_ephemeral_constant():
    import random
    terminals = [Symbol("x_0"), Symbol("x_1")]
    operators = [Ephemeral("c", random.random)]
    pset = create_pset(terminals + operators)

    MyClass = Cartesian("MyClass", pset)
    ind1 = MyClass([[2, 0]], [2])
    s1 = to_polish(ind1, return_args=False)
    s2 = to_polish(ind1.clone(), return_args=False)
    assert s1 != s2
    ind3 = point_mutation(ind1)
    assert not ind3.memory  # empty dict
    assert ind1.memory == pickle.loads(pickle.dumps(ind1)).memory


def test_structural_constant_cls(sc):
    assert 0.5 == sc.function("x", "f(x)")


def test_structural_constant_to_polish(sc):
    primitives = [Symbol("x_0"), sc]
    pset = create_pset(primitives)

    MyClass = Cartesian("MyClass", pset)
    ind = MyClass([[[1, 0, 0]], [[1, 0, 0]], [[1, 0, 0]]], [2])
    assert to_polish(ind, return_args=False) == ["1.0"]


@hypothesis.given(
    n_rows=integers(1, 5),
    n_columns=integers(1, 5),
    n_back=integers(1, 5),
    n_inputs=integers(1, 5),
    n_out=integers(1, 5))
def test__get_valid_inputs(n_rows, n_columns, n_back, n_inputs, n_out):

    valid_inputs = _get_valid_inputs(n_rows, n_columns, n_back, n_inputs,
                                     n_out)
    assert len(valid_inputs) == n_out + n_inputs + n_rows * n_columns

    for k, v in valid_inputs.items():
        assert all(i >= 0 for i in v)
        if k >= n_inputs:
            assert v
        else:
            assert not v


def test__get_valid_inputs_edge_case():
    valid_inputs = _get_valid_inputs(1, 2, 1, 1, 1)
    assert valid_inputs == {0: [], 1: [0], 2: [0, 1], 3: [0, 1, 2]}
