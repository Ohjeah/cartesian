import itertools
import copy
import sys
import re
from operator import attrgetter
from collections import namedtuple

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_random_state

from cartesian.util import make_it


class Primitive(object):
    """Base class"""
    def __init__(self, name, function, arity):
        """
        :param name: for text representation
        :param function: 
        :param arity: 
        """
        self.name = name
        self.function = function
        self.arity = arity


class Terminal(Primitive):
    """
    Base class for variables.
    Will always be used in the boilerplate ensuring a uniform signature, even if variable is not used in the genotype.
    """

    arity = 0

    def __init__(self, name):
        self.name = name


class Constant(Terminal):
    """
    Base class for symbolic constants.
    Will be used for constant optimization.
    Boilerplate: will only appear when used.
    """
    pass


class Ephemeral(Primitive):
    """
    Base class for ERC's.
    ERC's are terminals, but they are implemented as zero arity functions, because they do not need to appear in the 
    argument list of the lambda expression.
    
    .. Note ::
    
       Compilation behaviour: Each individual has a dict to store its numeric values. Each position in the code block will
       only execute the function once. Values are lost during copying.
    
    """
    def __init__(self, name, function):

        """
        :param name: for text representation
        :param function: callback(), should return a random numeric values.
        """
        super().__init__(name, function, 0)


class Structural(Primitive):
    """
    Structural constants are operators which take the graph representation of its arguments
    and convert it to a numeric value.
    """

    def __init__(self, name, function, arity):
        """
        :param name: for text representation 
        :param function: 
        :param arity: 
        """
        self.name = name
        self._function = function
        self.arity = arity

    def function(self, *args):
        return self._function(*map(self.get_len, args))

    @staticmethod
    def get_len(expr, tokens="(,"):
        """
        Get the length of a tree by parsing its polish notation representation
        :param expr: 
        :param tokens: to split at
        :return: length of expr 
        """
        regex = "|".join("\\{}".format(t) for t in tokens)
        return len(re.split(regex, expr))


PrimitiveSet = namedtuple("PrimitiveSet", "operators terminals max_arity mapping imapping context")


def create_pset(primitives):
    """Create immutable PrimitiveSet with some attributes for quick lookups"""
    terminals = [p for p in primitives if isinstance(p, Terminal)]
    operators = [p for p in primitives if p not in terminals]

    if operators:
        max_arity = max(operators, key=attrgetter("arity")).arity
    else:
        max_arity = 0

    mapping = {i: prim for i, prim in enumerate(sorted(terminals, key=attrgetter("name"))
                  + sorted(operators, key=attrgetter("name")))}

    imapping = {v: k for k, v in mapping.items()}
    context = {f.name: f.function for f in operators}

    return PrimitiveSet(operators=operators, terminals=terminals, imapping=imapping,
                        max_arity=max_arity, mapping=mapping, context=context)


def _make_map(*lists):
    i = 0
    for c, l in enumerate(lists):
        for r, el in enumerate(l):
            yield i, el, c, r, l
            i += 1


class Base(TransformerMixin):
    def __init__(self, code, outputs):
        self.inputs = list(range(len(self.pset.terminals)))
        self.code = code
        self.outputs = outputs
        # Primitives are allows to write their name values for storage
        self.memory = {}

    @property
    def map(self):
        return {i: (el, c, r, l) for i, el, c, r, l in _make_map(self.inputs, *self.code, self.outputs)}

    def __getitem__(self, index):
        return self.map[index][0]

    def __setitem__(self, index, item):
        el, c, r, l = self.map[index]
        l[r] = item

    def __len__(self):
        return max(self.map) + 1

    def __repr__(self):
        return "\n".join(to_polish(self, return_args=False))

    def __getstate__(self):
        # for compatibility with vanilla pickle protocol
        state = dict(self.__dict__)
        try:
            del state["_transform"]
        except KeyError:
            pass
        return state

    def __copy__(self):
        """save copy, discard memory to refresh random constants"""
        return type(self)(self.code[:], self.outputs[:])

    def clone(self):
        return copy.copy(self)

    def format(self, x):
        return "{}".format(x)

    def fit(self, x, y=None, **fit_params):
        self._transform = compile(self)
        self.fit_params = fit_params
        return self

    def transform(self, x, y=None):
        return self._transform(*x.T)

    @classmethod
    def create(cls, random_state=None):
        """
        Each gene is picked with a uniform distribution from all allowed inputs or functions.
        :param random_state: an instance of np.random.RandomState, a seed integer or None
        :return: A random new class instance.
        """
        random_state = check_random_state(random_state)

        operator_keys = list(range(len(cls.pset.terminals), max(cls.pset.mapping) + 1))
        code = []
        for i in range(cls.n_columns):
            column = []
            for j in range(cls.n_rows):
                min_input = max(0, (i-cls.n_back)*cls.n_rows) + len(cls.pset.terminals)
                max_input = i * cls.n_rows - 1 + len(cls.pset.terminals)
                in_ = list(range(min_input, max_input)) + list(range(0, len(cls.pset.terminals)))
                gene = [random_state.choice(operator_keys)] + [random_state.choice(in_) for _ in range(cls.pset.max_arity)]
                column.append(gene)
            code.append(column)
        outputs = [random_state.randint(0, cls.n_columns*cls.n_rows + len(cls.pset.terminals)) for _ in range(cls.n_out)]
        return cls(code, outputs)


class Cartesian(type):
    """
    Meta class to set class parameters and primitive set.
    """
    def __new__(mcs, name, primitive_set, n_columns=3, n_rows=1, n_back=1, n_out=1):
        dct = dict(pset=primitive_set, n_columns=n_columns, n_rows=n_rows, n_back=n_back, n_out=n_out)
        cls = super().__new__(mcs, name, (Base, ), dct)
        setattr(sys.modules[__name__], name, cls)
        return cls

    def __init__(cls, name, primitive_set, n_columns=3, n_rows=1, n_back=1, n_out=1):
        dct = dict(pset=primitive_set, n_columns=n_columns, n_rows=n_rows, n_back=n_back, n_out=n_out)
        super().__init__(name, (Base, ), dct)


def point_mutation(individual, random_state=None):
    """
    Randomly pick a gene in individual and mutate it.
    The mutation is either rewiring, i.e. changing the inputs, or changing the operator (head of gene)
    :param individual: instance of Base
    :type individual: instance of Cartesian
    :param random_state: an instance of np.random.RandomState, a seed integer or None
    :return: new instance of Base
    """
    random_state = check_random_state(random_state)
    n_terminals = len(individual.pset.terminals)
    i = random_state.randint(n_terminals, len(individual))
    el, c, r, l = individual.map[i]
    gene = l[r]
    if isinstance(gene, list):
        new_gene = gene[:]
        j = random_state.randint(0, len(gene))
        if j == 0:  # function
            new_j = individual.pset.imapping[random_state.choice(individual.pset.operators)]
        else:       # input
            min_input = max(0, (c - 1 - individual.n_back)*individual.n_rows + n_terminals)
            max_input = max(0, (c - 1) * individual.n_rows - 1 + n_terminals)
            in_ = list(range(min_input, max_input)) + list(range(n_terminals))
            new_j = random_state.choice(in_)
        new_gene[j] = new_j

    else: # output gene
        new_gene = random_state.randint(0, individual.n_columns*individual.n_rows + len(individual.inputs))
    new_individual = copy.copy(individual)
    new_individual[i] = new_gene
    return new_individual


def to_polish(c, return_args=True):
    """
    Return polish notation of expression encoded by c.
    Optionally return the used arguments.
    
    Resolve the outputs recursively.
    
    .. Note :: 
       Function has side-effects on the individual c.
       See Terminals for details
    
    :param c: instance of Base
    :type c: instance of Cartesian
    """
    primitives = c.pset.mapping
    used_arguments = set()

    def h(g):
        gene = make_it(c[g])
        primitive = primitives[next(gene)]

        # refactor to primitive.format() ? side-effects?
        if primitive.arity == 0:
            if isinstance(primitive, Terminal):
                used_arguments.add(primitive)

            elif isinstance(primitive, Ephemeral):
                if g not in c.memory:
                    c.memory[g] = c.format(primitive.function())
                return c.memory[g]

            if isinstance(primitive, Terminal):
                return primitive.name
            else:
                return "{}()".format(primitive.name)

        elif isinstance(primitive, Structural):
            return c.format(primitive.function(*[h(a) for a, _ in zip(gene, range(primitive.arity))]))

        else:
            return "{}({})".format(primitive.name,
                                   ", ".join(h(a) for a, _ in zip(gene, range(primitive.arity))))

    polish = [h(o) for o in c.outputs]

    if return_args:
        return polish, used_arguments
    else:
        return polish


def boilerplate(c, used_arguments=()):
    """
    Return the overhead needed to compile the polish notation.
    
    If used_arguments are provided, the boilerplate will only include
    the constants which are used as well as all variables.
    
    :param c: instance of Base
    :type c: instance of Cartesian
    :param used_arguments: list of terminals actually used in c.
    """
    mapping = c.pset.mapping
    if used_arguments:
        index = sorted([k for (k, v) in mapping.items() if v in used_arguments])
        args = [mapping[i] for i in index]
    else:
        args = [mapping[i] for i in c.inputs]
    args = [a for a in args if not isinstance(a, Constant)] + [a for a in args if isinstance(a, Constant)]
    return "lambda {}:".format(", ".join(a.name for a in args))


def compile(c):
    """
    :param c: instance of Base
    :type c: instance of Cartesian
    :return: lambda function
    """
    polish, args = to_polish(c, return_args=True)
    for t in c.pset.terminals:
        if not isinstance(t, Constant):
            args.add(t)
    bp = boilerplate(c, used_arguments=args)
    code = "({})".format(", ".join(polish)) if len(polish) > 1 else polish[0]
    return eval(bp + code, c.pset.context)
