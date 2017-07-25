from operator import attrgetter
from collections import namedtuple

import attr
from sklearn.utils.validation import check_random_state


@attr.s
class Primitive(object):
    name = attr.ib()
    function = attr.ib()
    arity = attr.ib()


@attr.s
class Terminal():
    arity = 0
    name = attr.ib()


class Constant(Terminal):
    pass


class ERC(Constant):
    def __init__(self, function):
        self.name = function()
        self.function = function

    def __copy__(self):
        return type(self)(self.function)


PrimitiveSet = namedtuple("PrimitiveSet", "operators terminals max_arity mapping")

def create_pset(primitives):
    operators = [p for p in primitives if p.arity > 0]
    terminals = [p for p in primitives if p.arity == 0]
    max_arity = max(operators, key=attrgetter("arity")).arity

    mapping = {i: prim for i, prim in enumerate(sorted(terminals, key=attrgetter("name")) \
                                              + sorted(terminals, key=attrgetter("name")))}

    return PrimitiveSet(operators=operators, terminals=terminals,
                        max_arity=max_arity, mapping=mapping)


@attr.s
class Cartesian:
    inputs = attr.ib()
    code = attr.ib()
    outputs = attr.ib()
    pset = None

    @classmethod
    def create(cls, n_in, n_columns, n_rows, n_back, n_out, random_state=None):
        random_state = check_random_state(random_state)

        operator_keys = list(range(len(cls.pset.terminals), max(cls.pset.mapping) + 1))
        inputs = [random_state.randint(0, len(cls.pset.terminals) + 1) for _ in range(n_out)]
        code = []
        for i in range(n_columns):
            column = []
            for j in range(n_rows):
                min_input = max(0, (i-n_back)*n_rows) + len(cls.pset.terminals)
                max_input = i * n_rows + j + len(cls.pset.terminals)
                inputs = list(range(min_input, max_input)) + list(range(0, len(cls.pset.terminals)))
                gene = [random_state.choice(operator_keys)] + [random_state.choice(inputs) for _ in range(cls.pset.max_arity)]
                column.append(gene)
            code.append(column)
        outputs = [random_state.randint(0, n_columns*n_rows + n_in) for _ in range(n_out)]

        return cls(inputs=inputs, code=code, outputs=outputs)


if __name__ == '__main__':
    terminals = [Terminal("x_0")]
    operators = [Primitive("f", lambda x: x, 1)]

    pset = create_pset(terminals + operators)

    Cartesian.pset = pset
    print(Cartesian.create(1, 2, 2, 1, 2))
