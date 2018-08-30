import copy
import itertools
import re
import sys
from operator import attrgetter

from dataclasses import dataclass
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_random_state

from cartesian.util import make_it


class Primitive(object):
    def __init__(self, name, function, arity):
        """Basic build block for cartesian programs.

        :param name: for text representation
        :param function:
        :param arity:
        """
        self.name = name
        self.function = function
        self.arity = arity


class Symbol(Primitive):
    def __init__(self, name):
        """Base class for variables.

        Will always be used in the boilerplate ensuring a uniform signature.
        Even if variable is not used in the genotype.

        Args:
            name: name of the primitive
        """
        self.name = name
        self.arity = 0


class Constant(Symbol):
    """Base class for symbolic constants.

    Will be used for constant optimization.
    Boilerplate: will only appear when used.
    """


class Ephemeral(Primitive):
    def __init__(self, name, fun):
        """Base class for ERC's.

        ERC's are terminals, but they are implemented as zero arity functions, as they do not need to appear in the
        argument list of the lambda expression.

        Note:
            Compilation behaviour: Each individual has a dict to store its numeric values.
            Each position in the code block will only execute the function once.
            Values are lost during copying.

        Args:
            name: for text representation
            fun: callback, should return a random numeric values.
        """
        super().__init__(name, fun, 0)


class Structural(Primitive):
    def __init__(self, name, fun, arity):
        """Structural constants are operators which take the graph representation of its arguments
            and convert it to a numeric value.

        Args:
            name: for text representation
            fun:
            arity:
        """
        self.name = name
        self._function = fun
        self.arity = arity

    def function(self, *args):
        return self._function(*map(self.get_len, args))

    @staticmethod
    def get_len(expr, tokens="(,"):
        """ Get the length of a tree by parsing its polish notation representation

        Args:
            expr: a formula in polish notation
            tokens: symbols to split the expression at

        Returns:
            length of expr

        """
        regex = "|".join("\\{}".format(t) for t in tokens)
        return len(re.split(regex, expr))


@dataclass
class PrimitiveSet:
    """A container holding the primitives and pre-compiled helper attributes.

    Args:
        operators: all non-terminal primitives (arity > 0)
        terminals: all terminals
        max_arity: maximum arity of all terminals. Determines the number of links for each register
        mapping: sorted and indexed list of the primitive set
        imapping: inverse of mapping
        context: links names of primitives to their functions
        symbols: all sybolic constants

    """

    operators: list
    terminals: list
    mapping: dict
    imapping: dict
    context: dict
    symbols: list
    max_arity: int

    @classmethod
    def create(cls, primitives):
        """Create immutable PrimitiveSet with some attributes for quick lookups"""
        terminals = [p for p in primitives if p.arity == 0]
        symbols = [p for p in primitives if isinstance(p, Symbol)]
        non_symbols = [p for p in terminals if not isinstance(p, Symbol)]
        operators = [p for p in primitives if p.arity > 0]
        if operators:
            max_arity = max(operators, key=attrgetter("arity")).arity
        else:
            max_arity = 0
        mapping = {
            i: prim
            for i, prim in enumerate(
                sorted(symbols, key=attrgetter("name"))
                + sorted(non_symbols, key=attrgetter("name"))
                + sorted(operators, key=attrgetter("name"))
            )
        }
        imapping = {v: k for k, v in mapping.items()}
        context = {f.name: f.function for f in operators}
        return cls(
            operators=operators,
            terminals=terminals,
            imapping=imapping,
            max_arity=max_arity,
            mapping=mapping,
            context=context,
            symbols=symbols,
        )


def _make_map(*lists):
    i = 0
    for c, l in enumerate(lists):
        for r, el in enumerate(l):
            yield i, el, c, r, l

            i += 1


def _code_index(n_in, n_row, c, r):
    return n_in + c * n_row + r


def _out_index(n_rows, n_columns, n_in, o):
    return n_rows * n_columns + n_in + o


def _get_valid_inputs(n_rows, n_columns, n_back, n_inputs, n_out):
    """dict of valid input genes for each gene in genom.
    Genes in the same column are not allowed to be connected to each other.
    """
    inputs = {i: [] for i in range(n_inputs)}
    for c in range(n_columns):
        first_index_this_column = c * n_rows + n_inputs
        min_ = max(min(first_index_this_column - n_rows * n_back, n_inputs), 0)
        max_ = first_index_this_column
        for r in range(n_rows):
            i = _code_index(n_inputs, n_rows, c, r)
            inputs[i] = list(range(min_, max_)) + list(range(n_inputs))
    min_ = min(max(0, (n_columns - n_back - 1) * n_rows), n_inputs)
    max_ = max(inputs)
    for o in range(n_out):
        inputs[o + max_ + 1] = list(range(min_, max_ + 1)) + list(range(n_inputs))
    for k, v in inputs.items():
        inputs[k] = list(set(v))
    return inputs


class Base(TransformerMixin):
    def __init__(self, code, outputs):
        self.n_inputs = len(self.pset.terminals)
        self.inputs = list(range(self.n_inputs))
        self.symbols = self.inputs[: len(self.pset.symbols)]
        self.code = code
        self.outputs = outputs
        # Primitives are allowed to write their name values for storage
        self.memory = {}

    @property
    def mapping(self):
        """Helper dictionary to index the cartesian registers."""
        return {i: (el, c, r, l) for i, el, c, r, l in _make_map(self.inputs, *self.code, self.outputs)}

    @property
    def _out_idx(self):
        """Returns the indices of the output genes in self"""
        return [_out_index(self.n_rows, self.n_columns, self.n_inputs, o) for o in range(self.n_out)]

    @property
    def active_genes(self):
        """Computes the set of active gene in an individual."""

        active = set(self._out_idx)  # all output genes are always active

        def h(g):
            active.add(g)
            gene = make_it(self[g])  # todo (mq): make this a method
            primitive = self.pset.mapping[next(gene)]
            for a, _ in zip(gene, range(primitive.arity)):
                h(a)

        for o in self.outputs:
            h(o)

        return active - set(self.inputs)

    def __getitem__(self, index):
        return self.mapping[index][0]

    def __setitem__(self, index, item):
        # element, column, row, gene
        *_, row, gene = self.mapping[index]
        gene[row] = item

    def __len__(self):
        """Returns the number of genes in self."""
        return self.n_columns * self.n_rows + self.n_out + self.n_inputs

    def __str__(self):
        return "\n".join(to_polish(self, return_args=False))

    def __repr__(self):  # pragma: no cover
        return f"{type(self)}\n\tInputs: {self.inputs}\n\tOutputs: {self.outputs}\n\tCode: {self.code}"

    def __getstate__(self):
        # for compatibility with vanilla pickle protocol
        state = dict(self.__dict__)
        try:
            del state["_transform"]
        except KeyError:
            pass
        return state

    def __copy__(self):
        """Save copy, discard memory to refresh random constants"""
        return type(self)([c[:] for c in self.code], self.outputs[:])

    def clone(self):
        """Save copy, discard memory to refresh random constants"""
        return copy.copy(self)

    @staticmethod
    def format(x):  # pragma: no cover
        return "{}".format(x)

    def fit(self, x, y=None, **fit_params):
        self._transform = compile(self)
        self.fit_params = fit_params
        return self

    def transform(self, x, y=None):
        return self._transform(*x.T)

    @classmethod
    def create(cls, random_state=None):
        """Creates a new individual.

        Each gene is picked with a uniform distribution from all allowed inputs or functions.

        Args:
            random_state: an instance of np.random.RandomState, a seed integer or None

        Returns:
            a new (random) individual

        """
        random_state = check_random_state(random_state)
        n_in = len(cls.pset.terminals)
        operator_keys = list(range(n_in, max(cls.pset.mapping) + 1))
        code = []
        for i in range(cls.n_columns):
            column = []
            for j in range(cls.n_rows):
                index = _code_index(n_in, cls.n_rows, i, j)
                in_ = cls._valid_inputs[index]
                gene = [random_state.choice(operator_keys)] + [
                    random_state.choice(in_) for _ in range(cls.pset.max_arity)
                ]
                column.append(gene)
            code.append(column)
        outputs = [
            random_state.choice(cls._valid_inputs[_out_index(cls.n_rows, cls.n_columns, n_in, o)])
            for o in range(cls.n_out)
        ]
        return cls(code, outputs)


class Cartesian(type):
    """Meta class to set class parameters and primitive set."""

    def __new__(mcs, name, primitive_set, n_columns=3, n_rows=1, n_back=1, n_out=1):
        valid_inputs = _get_valid_inputs(n_rows, n_columns, n_back, len(primitive_set.terminals), n_out)
        dct = dict(
            pset=primitive_set,
            n_columns=n_columns,
            n_rows=n_rows,
            n_back=n_back,
            n_out=n_out,
            _valid_inputs=valid_inputs,
        )
        cls = super().__new__(mcs, name, (Base,), dct)
        setattr(sys.modules[__name__], name, cls)
        return cls

    def __init__(cls, name, primitive_set, n_columns=3, n_rows=1, n_back=1, n_out=1):
        valid_inputs = _get_valid_inputs(n_rows, n_columns, n_back, len(primitive_set.terminals), n_out)
        dct = dict(
            pset=primitive_set,
            n_columns=n_columns,
            n_rows=n_rows,
            n_back=n_back,
            n_out=n_out,
            _valid_inputs=valid_inputs,
        )
        super().__init__(name, (Base,), dct)


def _point_mutation(individual, idx, random_state):
    """Apply a point mutation on individual at position idx

    Args:
        individual: instance of Base
        idx: index
        random_state: an instance of np.random.RandomState

    Returns:
        mutated individual
    """
    el, c, r, lst = individual.mapping[idx]
    gene = lst[r]

    if isinstance(gene, list):
        new_gene = gene[:]
        j = random_state.randint(0, len(gene))
        if j == 0:  # function
            choices = individual.pset.operators[:]
            if len(choices) > 1:  # don't allow the same function again
                prim = individual.pset.mapping[gene[j]]
                choices.pop(choices.index(prim))
            new_j = individual.pset.imapping[random_state.choice(choices)]
        else:  # input
            valid_inputs = individual._valid_inputs[idx][:]
            if len(valid_inputs) > 1:  # don't allow the same inputs again
                valid_inputs.pop(valid_inputs.index(gene[j]))
            new_j = random_state.choice(valid_inputs)
        new_gene[j] = new_j
    else:  # output gene
        new_gene = random_state.randint(0, len(individual) - individual.n_out - 1)
    new_individual = individual.clone()
    new_individual[idx] = new_gene
    return new_individual


def mutate(individual, n_mutations=1, method="active", random_state=None):
    """Create offsprings by mutating an individual.

    Args:
        individual: instance of Base
        n_mutations: number of mutations
        method: specific mutation method
        random_state: an instance of np.random.RandomState, a seed integer or None

    Returns:
        mutated individual

    """
    random_state = check_random_state(random_state)
    mutate_ = active_gene_mutation if method == "active" else point_mutation
    offspring = individual.clone()
    for _ in range(n_mutations):
        offspring = mutate_(offspring, random_state=random_state)
    return offspring


def active_gene_mutation(individual, random_state=None):
    """Picks an active gene and

    Args:
        individual: instance of Base
        random_state: an instance of np.random.RandomState, a seed integer or None

    Returns:
        mutated individual
    """
    random_state = check_random_state(random_state)
    i = random_state.choice(list(individual.active_genes), 1)[0]
    return _point_mutation(individual, i, random_state)


def point_mutation(individual, random_state=None):
    """Picks a gene at random and mutates it.

    The mutation is either rewiring, i.e. changing the inputs, or changing the operator (head of gene).

    Args:
        individual: instance of Base
        random_state: an instance of np.random.RandomState, a seed integer or None

    Returns:
        mutated individual
    """
    random_state = check_random_state(random_state)
    n_terminals = len(individual.pset.terminals)
    i = random_state.randint(n_terminals, len(individual) - 1)
    return _point_mutation(individual, i, random_state)


def to_polish(c, return_args=True):
    """Generates the polish notation of expression encoded by c.

    Resolves the outputs recursively.

    Note:
       Function has side-effects on the individual c.
       See Symbols for details.

    Args:
        c: instance of base
        return_args: optionally return the used arguments too

    Returns:
        polish notation of expression encoded by c
    """
    primitives = c.pset.mapping
    used_arguments = set()

    def h(g):
        gene = make_it(c[g])
        primitive = primitives[next(gene)]
        # refactor to primitive.format() ? side-effects?
        if primitive.arity == 0:
            if isinstance(primitive, Symbol):
                used_arguments.add(primitive)
            elif isinstance(primitive, Ephemeral):
                if g not in c.memory:
                    c.memory[g] = c.format(primitive.function())
                return c.memory[g]

            if isinstance(primitive, Symbol):
                return primitive.name

            else:
                return "{}()".format(primitive.name)

        elif isinstance(primitive, Structural):
            return c.format(primitive.function(*[h(a) for a, _ in zip(gene, range(primitive.arity))]))

        else:
            return "{}({})".format(
                primitive.name, ", ".join(h(a) for a, _ in zip(gene, range(primitive.arity)))
            )

    polish = [h(o) for o in c.outputs]
    if return_args:
        return polish, used_arguments

    else:
        return polish


def _boilerplate(c, used_arguments=()):
    """Generates the overhead needed to compile the polish notation.

    If used_arguments are provided, the boilerplate will only include
    the constants which are used as well as all variables.

    Args:
        c: instance of Base
        used_arguments: list of terminals actually used in c.

    Returns:
        overhead needed to compile the polish notation.

    """
    mapping = c.pset.mapping
    if used_arguments:
        index = sorted([k for (k, v) in mapping.items() if v in used_arguments])
        args = [mapping[i] for i in index]
    else:
        args = [mapping[i] for i in c.inputs]
    args_ = [a for a in args if isinstance(a, Symbol) and not isinstance(a, Constant)]
    args_ += [a for a in args if isinstance(a, Constant)]
    return "lambda {}:".format(", ".join(a.name for a in args_))


def compile(c):
    """Transform an individual into a lambda function

    Args:
        c: instance of Base

    Returns:
        lambda function

    """
    polish, args = to_polish(c, return_args=True)
    for t in c.pset.symbols:
        if not isinstance(t, Constant):
            args.add(t)
    bp = _boilerplate(c, used_arguments=args)
    code = "({})".format(", ".join(polish)) if len(polish) > 1 else polish[0]
    return eval(bp + code, c.pset.context)
