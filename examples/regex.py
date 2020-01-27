import re
import random
from cartesian.cgp import *
from cartesian.algorithm import oneplus


def group(s):
    trans = str.maketrans({"[": "", "]": "", "*": "", "+": "", "$": "", "^": ""})
    return f"[{s.translate(trans)}]"


def or_(a, b):
    if a == b:
        return a
    return f"{a}|{b}"


def add(a, b):
    return a.replace("$", "") + b.replace("^", "")


primitives = [
    Primitive("add", add, 2),
    Primitive("star", lambda s: f"{s}*".replace("**", "*").replace("+*", "*"), 1),
    Primitive("plus", lambda s: f"{s}+".replace("++", "+").replace("*+", "+"), 1),
    Primitive("group", group, 1),
    Primitive("or", or_, 2),
    # Primitive("begin", lambda s: f"^{s.replace('^','')}", 1),
    # Primitive("end", lambda s: f"{s.replace('$', '')}$", 1),
    Symbol("a"),
    Symbol("b"),
    Symbol("."),
]
pset = PrimitiveSet.create(primitives)

MyCartesian = Cartesian("MyCartesian", pset, n_rows=1, n_columns=10, n_out=1, n_back=1)


def compile(program):
    stack = []
    out = program._out_idx[0]
    for _, node in reversed(list(program._iter_subgraph(program[out]))):
        if node.arity > 0 and node.arity <= len(stack):
            args = [stack.pop() for _ in range(node.arity)]
            stack.append(node.function(*args))
        elif node.arity == 0:
            stack.append(node.name)
        else:
            raise ValueError("Incorrect program.")
    return re.compile(stack[-1])


data = {
    k: -1 if "aba" in k else 1
    for k in map(lambda x: "".join(random.sample("aaaaabbbbb", k=random.randint(1, 7))), range(20))
}

print(data, sum(v for v in data.values() if v == -1))


def evaluate(program):
    regex = compile(program)
    return sum(v for k, v in data.items() if re.match(regex, k))


res = oneplus(evaluate, cls=MyCartesian, maxiter=10000, n_jobs=1, f_tol=-1000000)
print(res)
print(compile(res.ind))
