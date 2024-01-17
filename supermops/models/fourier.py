import itertools

import numpy as np

from .abstract import ForwardModel


class FourierNDModel(ForwardModel):
    def __init__(self, freqs, param_max=None):
        """Fourier measurements for given frequencies.

        Arguments:
            freqs {ndarray} -- array [freqs1, ..., freqsn], where n is the
                number of parameters and freqsi is an array containing all
                frequencies to be measured in the ith dimension
            param_max {ndarray} -- upper bounds on parameters
        """
        if param_max is None:
            param_max = np.ones(len(freqs))
        assert len(freqs) == len(param_max)

        self.param_dim = len(freqs)
        self.freqs = np.asarray(freqs)
        self.param_max = np.asarray(param_max)
        self.freq_grid = np.array(list(itertools.product(*self.freqs)),
                                  dtype=np.float64)

    def apply_single_source(self, source):
        complex_vals = np.exp(-2j * np.pi *
                              (self.freq_grid @ (source / self.param_max)))
        return np.hstack((np.real(complex_vals), np.imag(complex_vals)))

    def apply_multiple(self, sources):
        complex_vals = np.exp(-2j * np.pi *
                              (self.freq_grid @ (sources.transpose() / self.param_max[:,None])))
        return np.vstack((np.real(complex_vals), np.imag(complex_vals)))

    def get_param_bounds(self):
        return np.array([np.zeros(self.param_dim), self.param_max])

    def meas_size(self):
        return 2 * len(self.freq_grid)
