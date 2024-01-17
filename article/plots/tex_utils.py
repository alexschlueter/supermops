def lists_to_dicts(lists, keys):
    return map(lambda vals: dict(zip(keys, vals)), zip(*lists))

def to_coord(num, acc=3):
    return f"{num:.{acc}f}"

def iter_to_coords(it, acc=3, sep=","):
    return sep.join([to_coord(i, acc) for i in it])

def iter_to_path(it, cs=""):
    if cs:
        cs += ":"
    return " -- ".join(map(lambda c: f"({cs}{iter_to_coords(c)})", it))