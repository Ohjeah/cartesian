def make_it(x):
    """
    Ensure that x is an iterator.
    If x is not iterable, wrap it as a one-elemened tuple.
    """
    try:
        return iter(x)
    except TypeError:
        x = x,
        return iter(x)
