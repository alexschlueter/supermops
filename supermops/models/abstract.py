from abc import ABC, abstractmethod

import numpy as np


class ForwardModel(ABC):
    """An abstract forward model which implements a linear forward operator
    by calculating measurements of source points.
    """
    @abstractmethod
    def apply_single_source(self, source):
        """Calculate measurement for a single source point.

        Arguments:
            source {ndarray} -- source point

        Returns:
            ndarray -- resulting measurement
        """
        pass

    @abstractmethod
    def get_param_bounds(self):
        """Get the bounds for all parameters in the format
        [min_bounds, max_bounds], where both are arrays of length equal to
        the number of parameters of the model.
        """
        pass

    @abstractmethod
    def meas_size(self):
        """Returns the dimension of the measurements (length of the
        arrays returned by apply).
        """
        pass

    def apply(self, support, weights):
        """Calculate the measurement of a linear combination of source points
        with corresponding weights.

        Arguments:
            support {ndarray} -- array of source points
            weights {ndarray} -- weights for each source point

        Returns:
            ndarray -- measurement result
        """
        return np.sum(
            [
                weight * self.apply_single_source(source)
                for weight, source in zip(weights, support)
            ],
            axis=0,
        )
