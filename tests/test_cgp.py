import operator
import pathlib
import pickle
import sys

import flaky
import hypothesis
import hypothesis.strategies as s
import numpy as np
import pytest
from hypothesis.strategies import builds
from hypothesis.strategies import integers

from cartesian.cgp import *
from cartesian.cgp import _boilerplate
from cartesian.cgp import _get_valid_inputs

sys.path.append(pathlib.Path(__name__).parent.as_posix())
from conftest import pset


def make_ind(random_state=None, **kwargs):
    return Cartesian(**kwargs).create(random_state=random_state)


ind_strat = builds(
    make_ind,
    name=s.just("Individual"),
    primitive_set=s.just(pset()),
    n_columns=integers(min_value=1, max_value=10),
    n_rows=integers(min_value=1, max_value=10),
    n_back=integers(min_value=1, max_value=10),
    n_out=integers(min_value=1, max_value=5),
)


def test_PrimitiveSet(pset):
    assert pset.mapping == {
        1: pset.terminals[0],
        2: pset.terminals[1],
        0: pset.terminals[2],
        3: pset.operators[0],
    }
    assert pset.max_arity == 1
    assert pset.context[pset.operators[0].name] == operator.neg


class TestIndividual:
    def test_active(self, individual):
        assert individual.active_genes == {3, 4}

    def test__out_idx(self, individual):
        assert individual._out_idx == [4]

    def test_copy(self, individual):
        individual.memory[0] = 1
        new = individual.clone()
        with pytest.raises(KeyError):
            new.memory[0]
        assert new.code == individual.code
        assert new.code is not individual.code
        assert new.outputs == individual.outputs
        assert new.outputs is not individual.outputs

    def test_pickle(self, individual):
        pickled = pickle.loads(pickle.dumps(individual))
        for k in individual.__dict__.keys():
            assert pickled.__dict__[k] == individual.__dict__[k]

    def test_fit_transform(self, individual):
        x = np.ones((1, 2))
        y = individual.fit_transform(x)
        assert y == np.array([-1])

    def test_get(self, individual):
        assert individual[0] == 0
        assert individual[1] == 1

    def test_set(self, individual):
        n = len(individual)
        individual[n - 1] = 1
        assert individual.outputs[0] == 1

    def test_to_polish(self, individual):
        polish, used_arguments = to_polish(individual)
        assert polish == ["neg(x_0)"]
        assert len(used_arguments) == 1

    def test_boilerplate(self, individual):
        assert _boilerplate(individual) == "lambda x_0, x_1, c:"
        assert _boilerplate(individual, used_arguments=[individual.pset.terminals[0]]) == "lambda x_0:"

    def test_compile(self, individual):
        f = compile(individual)
        assert f(1, 1) == -1


def assert_different_individuals(old, new):
    assert new.inputs is not old.inputs
    assert new.inputs == old.inputs
    assert new.code is not old.code
    assert new.outputs is not old.outputs
    changes = 0
    if new.outputs != old.outputs:
        changes += 1
    if old.code != new.code:
        changes += 1
    assert changes > 0


@hypothesis.settings(max_examples=25)
@hypothesis.given(ind_strat, s.data())
def test_mutation(ind, data):
    new_ind = mutate(ind, n_mutations=5, method=data.draw(s.sampled_from(["active", "point"])), random_state=0)
    assert_different_individuals(ind, new_ind)


@flaky.flaky(50, 25)
def test_active_gene_mutation(individual):
    # active gene mutation can randomly produce the same individual again
    assert str(individual) != str(active_gene_mutation(individual))


def test_ephemeral_constant():
    import random

    terminals = [Symbol("x_0"), Symbol("x_1")]
    operators = [Ephemeral("c", random.random)]
    pset = PrimitiveSet.create(terminals + operators)
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
    pset = PrimitiveSet.create(primitives)
    MyClass = Cartesian("MyClass", pset)
    ind = MyClass([[[1, 0, 0]], [[1, 0, 0]], [[1, 0, 0]]], [2])
    assert to_polish(ind, return_args=False) == ["1.0"]


@hypothesis.given(
    n_rows=integers(1, 5),
    n_columns=integers(1, 5),
    n_back=integers(1, 5),
    n_inputs=integers(1, 5),
    n_out=integers(1, 5),
)
def test__get_valid_inputs(n_rows, n_columns, n_back, n_inputs, n_out):
    valid_inputs = _get_valid_inputs(n_rows, n_columns, n_back, n_inputs, n_out)
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
