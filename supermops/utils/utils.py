import builtins
import itertools
import pickle
import time

import numpy as np
from scipy import spatial

try:
    # https://github.com/pyutils/line_profiler
    @profile
    def test():
        pass
except NameError:
    # print("Defining dummy @profile...")
    def profile(func):
        return func
    builtins.profile = profile

class TimeInfo:
    def __init__(self, times):
        self.times = sorted(times)
        self.num_times = len(self.times)
        self.tmin = self.times[0]
        self.tmax = self.times[-1]
        self.tmid = (self.tmin + self.tmax) / 2
        self.tspan = self.tmax - self.tmin

    def centered(self, t):
        return t - self.tmid

def move(phase_points, time):
    num_sources = len(phase_points)
    phase_points = phase_points.reshape(num_sources, 2, -1)
    return phase_points[:,0] + time * phase_points[:,1]

def torus_dist(v, w):
    return np.max(np.minimum(np.abs(v - w), 1 - np.abs(v - w)))

def min_torus_sep(sources):
    if len(sources) < 2:
        return np.inf
    else:
        return np.min(spatial.distance.pdist(sources, torus_dist))

def min_dyn_sep(support, K=None, dt=None, times=None):
    """Given a list of moving particles, calculate the minimum separation in the
    torus distance on [0,1]^d between any two particles over the given
    time steps.

    Args:
        support: List of particles, each with position and velocity
        Time steps specified either with arguments K and dt or with times array.
    """
    if K is not None and dt is not None:
        times = np.linspace(-K * dt, K * dt, 2 * K + 1)
    min_seps = []
    for t in times:
        min_seps.append(min_torus_sep(move(support, t)))
    return np.min(min_seps)

def announce_save(path, obj, descr=None, with_time=False):
    path = str(path)
    if with_time:
        path += "_" + time.strftime("%Y%m%d-%H%M%S")
    if not path.endswith(".npy"):
        path += ".npy"
    if descr is not None:
        print("Saving {} to {}".format(descr, path))
    else:
        print("Saving {}".format(path))
    np.save(path, obj)

def announce_pickle(path, obj, descr=None, with_time=False):
    path = str(path)
    if with_time:
        path += "_" + time.strftime("%Y%m%d-%H%M%S")
    if not path.endswith(".pickle"):
        path += ".pickle"
    if descr is not None:
        print("Saving {} to {}".format(descr, path))
    else:
        print("Saving {}".format(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def int_to_str_ranges(data, reduction=None):
    """
    Represent a list of integers as a string, expressing consecutive integers
    as ranges
    """
    res = []
    for k, g in itertools.groupby(enumerate(data), lambda x: x[0]-x[1]):
        g = list(g)
        start, end = g[0][1], g[-1][1]
        if start == end:
            idstr = str(start)
        else:
            idstr = f"{g[0][1]}-{g[-1][1]}"
        if reduction is not None:
            jobs = [1 + (i - 1) // reduction for i in range(start, end + 1)]
            idstr += f" ({int_to_str_ranges(jobs)})"
        res.append(idstr)
    return ",".join(res)

def parse_int_set(instr):
    res = set()
    for tok in instr.split(","):
        if tok:
            try:
                res.add(int(tok))
            except ValueError:
                start, end = tok.split("-")
                res.update(range(int(start), int(end) + 1))
    return res

def dirichlet(grid, support, weights, fc):
    freqs = np.arange(-fc, fc + 1)
    res = 0
    for s, w in zip(support, weights):
        xx = np.outer(grid - s[0], freqs)
        xy = np.outer(grid - s[1], freqs)
        zx = np.exp(-2j * np.pi * xx).sum(axis=1)
        zy = np.exp(-2j * np.pi * xy).sum(axis=1)
        res += w * np.outer(zx, zy)
    return res
