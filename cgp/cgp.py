import itertools
from operator import attrgetter
from collections import namedtuple

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_random_state

from cgp.util import make_it


class Primitive(object):
    def __init__(self, name, function, arity):
        self.name = name
        self.function = function
        self.arity = arity


class Constant(Primitive):
    arity = 0
    def __init__(self, name, function=None):
        self.name = name
        self.function = function


class ERC(Primitive):
    arity = 0


class Terminal(Primitive):
    arity = 0
    def __init__(self, name):
        self.name = name


PrimitiveSet = namedtuple("PrimitiveSet", "operators terminals max_arity mapping context")


def create_pset(primitives):
    operators = [p for p in primitives if p.arity > 0]
    terminals = [p for p in primitives if p.arity == 0]

    if operators:
        max_arity = max(operators, key=attrgetter("arity")).arity
    else:
        max_arity = 0

    mapping = {i: prim for i, prim in enumerate(sorted(terminals, key=attrgetter("name")) \
                                              + sorted(operators, key=attrgetter("name")))}

    context = {f.name: f.function for f in operators}

    return PrimitiveSet(operators=operators, terminals=terminals,
                        max_arity=max_arity, mapping=mapping, context=context)

def _make_map(*lists):
    i = 0
    for l in lists:
        for j, el in enumerate(l):
            yield i, el, j, l
            i += 1

class Base(TransformerMixin):
    def __init__(self, code, outputs):
        self.inputs = list(range(len(self.pset.terminals)))
        self.code = code
        self.outputs = outputs

    @property
    def map(self):
        return {i: (el, j, l) for i, el, j, l in _make_map(self.inputs, *self.code, self.outputs)}

    def __getitem__(self, index):
        return self.map[index][0]

    def __setitem__(self, index, item):
        el, j, l = self.map[index]
        l[j] = item

    def __len__(self):
        return max(self.map) + 1

    def fit(self, x, y=None, **fit_params):
        self._transform = compile(self)
        self.fit_params = fit_params
        return self

    def transform(self, x, y=None):
        return self._transform(*x.T)

    @classmethod
    def create(cls, n_in, n_columns, n_rows, n_back, n_out, random_state=None):
        random_state = check_random_state(random_state)

        operator_keys = list(range(len(cls.pset.terminals), max(cls.pset.mapping) + 1))
        code = []
        for i in range(n_columns):
            column = []
            for j in range(n_rows):
                min_input = max(0, (i-n_back)*n_rows) + len(cls.pset.terminals)
                max_input = i * n_rows - 1 + len(cls.pset.terminals)
                in_ = list(range(min_input, max_input)) + list(range(0, len(cls.pset.terminals)))
                gene = [random_state.choice(operator_keys)] + [random_state.choice(in_) for _ in range(cls.pset.max_arity)]
                column.append(gene)
            code.append(column)
        outputs = [random_state.randint(0, n_columns*n_rows + n_in) for _ in range(n_out)]
        return cls(code, outputs)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_transform"]
        return state

def point_mutation(individual, random_state=None):
    random_state = check_random_state(random_state)

    i = random_state.randint(len(individual.pset.terminals), len(individual))
    gene

# class Cartesian(type):
#     def __new__(mcs, name, primitive_set):
#         print("new")
#         import sys
#         from inspect import getframeinfo, getmodulename, stack
#         cls = super().__new__(mcs, name, (Base,), {"pset": primitive_set})
#         caller = getframeinfo(stack()[1][0])         # find current_module by looking up caller in stack
#         filename = getmodulename(caller.filename)
#         try:
#             current_module = [mod for mname, mod in sys.modules.items() if filename == mname.split('.')[-1]][0]
#         except:
#             current_module = sys.modules["__main__"]
#         setattr(current_module, name, cls)
#         return getattr(current_module, name)
#
#
#     def __init__(cls, name, primitive_set):
#         print("init")
#         return super().__init__(name, (Base,), {"pset": primitive_set})


def to_polish(c, return_args=True):
    primitives = c.pset.mapping
    used_arguments = set()

    def h(g):
        gene = make_it(c[g])
        primitive = primitives[next(gene)]

        if primitive.arity == 0:
            if isinstance(primitive, Terminal):
                used_arguments.add(primitive)
            return primitive.name
        else:
            return "{}({})".format(primitive.name,
                                   ", ".join(h(a) for a, _ in zip(gene, range(primitive.arity))))

    polish = [h(o) for o in c.outputs]

    if return_args:
        return polish, used_arguments
    else:
        return polish


def boilerplate(c, used_arguments=()):
    mapping = c.pset.mapping
    if used_arguments:
        index = sorted([k for (k, v) in mapping.items() if v in used_arguments])
        args = [mapping[i] for i in index]
    else:
        args = [mapping[i] for i in c.inputs]
    args = [a for a in args if not isinstance(a, Constant)] + [a for a in args if isinstance(a, Constant)]
    return "lambda {}:".format(", ".join(a.name for a in args))


def compile(c):
    polish = to_polish(c, return_args=False)
    bp = boilerplate(c)
    code = "({})".format(", ".join(polish)) if len(polish) > 1 else polish[0]
    return eval(bp + code, c.pset.context)
