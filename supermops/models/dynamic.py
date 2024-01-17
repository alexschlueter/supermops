import numpy as np

from .abstract import ForwardModel


class DynamicSuperresModel(ForwardModel):
    def __init__(self, static_model, K=None, dt=None, times=None):
        """Add dynamic component to a given static model for discretized time
        steps. The measurement consists of the concatenation of the measurements
        of the static model at each time step.

        Arguments:
            static_model {ForwardModel} -- the forward model to apply at each
                time step
            K {int} -- time steps range from -K to K
            dt {float} -- time span between adjacent steps
        """
        self.param_dim = 2 * static_model.param_dim
        self.static_model = static_model
        if times is not None:
            assert K is None  and dt is None
            self.times = times
        else:
            assert K is not None and dt is not None
            self.times = np.linspace(-dt * K, dt * K, 2 * K + 1)

    def apply_single_source(self, source):
        return np.hstack([
            self.static_model.apply_single_source(
                source[:self.static_model.param_dim] +
                t * source[self.static_model.param_dim:]) for t in self.times
        ])

    def get_param_bounds(self):
        stat_bnds = self.static_model.get_param_bounds()
        v_bnds = [
            np.full(self.static_model.param_dim, -self.v_max),
            np.full(self.static_model.param_dim, self.v_max),
        ]
        return np.hstack([stat_bnds, v_bnds])

    def meas_size(self):
        return len(self.times) * self.static_model.meas_size()
