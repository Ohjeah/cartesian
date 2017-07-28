def make_it(x):
    try:
        return iter(x)
    except TypeError:
        x = x,
        return iter(x)
