import numpy as np

from .utils import min_dyn_sep


class SeparationDistribution:
    def __init__(self, generator, K, dt, separations, sep_range, num_bins=500):
        self.gen = generator
        self.K = K
        self.dt = dt
        self.hist, self.bins = np.histogram(separations,
                                            num_bins,
                                            range=sep_range,
                                            density=True)
        self.min_density = np.min(self.hist)

    def rejection_sampling(self, verbose=False):
        while True:
            support, weights = self.gen()
            sep = min_dyn_sep(support, self.K, self.dt)
            idx = np.digitize(sep, self.bins) - 1
            if 0 <= idx < len(self.hist):
                density = self.hist[idx]
                reject = np.random.rand()
                if reject < self.min_density / density:
                    return support, weights
                elif verbose:
                    print("Reject! {:.3f}>={:.3f}/{:.3f}".format(
                        reject, self.min_density, density))
            elif verbose:
                print("Reject! Sep {:.3f} out of range".format(sep))


def gen_separation_distr(generator, K, dt, num=10000, verbose=False):
    seps = []
    for it in range(num):
        if verbose and it % 1000 == 0:
            print("gen_separation_distr: it {}".format(it))
        support, _ = generator()
        sep = min_dyn_sep(support, K, dt)
        seps.append(sep)

    return seps
    # return SeparationDistribution(generator, K, dt, seps, sep_range, num_bins)
