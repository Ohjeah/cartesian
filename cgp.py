import operator

class Primitive():
    def __init__(self, name, function   ):
        self.name = name
        self.function = function

    @property
    def arity(self):
        return self.function.__code__.co_argcount


class Terminal():
    arity = 0
    def __init__(self, name):
        self.name = name


class Constant(Terminal):
    pass


class PrimitiveSet():
    def __init__(self, primitives):
        self.primitives = primitives
        self.memory = {i:p.name for i, p in enumerate(self.primitives)}

    @property
    def constants(self):
        pass

    @property
    def terminals(self):
        return [p for p in self.primitives if not p.arity]


add = Primitive("add", operator.add)
sub = Primitive("sub", operator.sub)
mul = Primitive("mul", operator.mul)
div = Primitive("div", operator.truediv)

x = Terminal("x")
c = Constant("c")

pset = PrimitiveSet([add, sub, mul, div, x , c])

print(pset.terminals)
exit()

from itertools import islice, chain
from toolz import last


class Cartesian():
    primitives = primitives

    def __getitem__(self, i):
        return last(islice(chain.from_iterable([self.inputs, *self.rows, self.outputs]), i+1))

    def in_inputs(self, i):
        return i < len(self.inputs)


def to_polish(c, return_args=True):
    primitives = c.primitives

    used_arguments = set()

    def h(g):
        n = iter(c[int(g[0])])
        f = int(next(n))
        fs = primitives[f]
        args = ", ".join(h(i) for i in n)
        if args:
            return fs + "(" + args + ")"
        else:
            used_arguments.add(fs)
            return fs

    polish = [h(g) for g in c.outputs]

    if return_args:
        return polish, used_arguments
    else:
        return polish


def boilerplate(c, used_arguments=()):
    primitives = c.primitives
    if used_arguments:
        index = sorted([k for (k, v) in primitives.items() if v in used_arguments])
        args = (primitives.get(i) for i in index)
    else:
        args = (c.primitives[int(i)] for i in c.inputs)
    return "lambda " + ", ".join(args) + ": "


def compile(c):
    code_ = to_polish(c, return_args=False)
    bp = boilerplate(c)
    code = "(" + ", ".join(code_) + ")" if len(code_) > 1 else code_[0]
    return eval(bp + code, context)


c = Cartesian()
c.inputs = ["4", "5"]
c.rows = [["001", "100", "131", "201", "044", "254"]]
c.outputs = ["2", "2", "1"]

print(to_polish(c))
f = compile(c)
print(f(1, 1))
