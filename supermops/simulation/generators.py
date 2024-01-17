import numpy as np


def two_passing(dist=0.06, vel=0.2):
    support = np.array([[0.5, 0.5 - dist / 2, vel, 0.0],
                        [0.5, 0.5 + dist / 2, -vel, 0.0]])
    weights = np.array([1.0, 1.0])
    return support, weights


def uniform_stay_in_square(K,
                           dt,
                           min_num_srcs=4,
                           max_num_srcs=20,
                           wmin=0.9,
                           wmax=1.1):
    """
    Randomly generate between min_num_srcs and max_num_srcs moving particles
    such that they stay in the square [0,1]^2 over the given time steps. Also
    generate weights between wmin and wmax.
    """
    num_sources = np.random.randint(min_num_srcs, max_num_srcs + 1)
    support = np.empty((num_sources, 4))
    for i in range(num_sources):
        while True:
            pos = np.random.rand(2)
            vel = (np.random.rand(2) - 0.5) / (K * dt)
            before = pos - K * dt * vel
            after = pos + K * dt * vel
            if np.all((0 < before) & (0 < after) & (before < 1) & (after < 1)):
                support[i] = np.hstack((pos, vel))
                break

    weights = wmin + np.random.rand(num_sources) * (wmax - wmin)
    return support, weights
