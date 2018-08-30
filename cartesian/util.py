import numpy as np


def make_it(x):
    """Ensures that x is an iterator.

    If x is not iterable, wrap it as a one-elemened tuple.
    """
    try:
        return iter(x)

    except TypeError:
        x = (x,)
        return iter(x)


@np.vectorize
def replace_nan(x, rep=np.infty):
    """Replace any nan in x with rep."""
    return rep if np.isnan(x) else x
