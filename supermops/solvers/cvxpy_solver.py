import cvxpy as cp
import numpy as np

class CVXPySolver:
    """
    Formulate the problem using the library CVXPY and solve it using the MOSEK
    backend. The problem is formulated as

    min_(mu, nu) ||mu||_1 + ||nu||_1 + data_scale * ||meas_block @ nu - data||_2^2
    s.t. ||redcons_block @ (mu, nu)|| <= redcons_bound
    """
    def __init__(self, space, redcons_block, meas_block, redcons_bound, data_scale=None, paper_alpha=None):
        """
        Args:
            space: ReducedLinearMotionSpace
            redcons_block: Matrix for reduced time consistency constraint
            meas_block: Measurement matrix (only applied to nu)
            redcons_bound: Scalar to use as bound in time consistency constraint
            data_scale: Scalar to multiply with data term. Give either this or
                        paper_alpha
            paper_alpha: Parameter from the paper, used as
                         data_scale = 1 / (2 * paper_alpha)
        """
        self.space = space

        self.cons_bound_param = cp.Parameter(nonneg=True, value=redcons_bound)
        # self.data_scale_param = cp.Parameter(nonneg=True, value=data_scale)
        if (data_scale is not None and paper_alpha is not None) or data_scale == paper_alpha == None:
            raise "either data_scale or paper_alpha"
        if data_scale is not None:
            self.data_scale_param = data_scale
        else:
            self.data_scale_param = 1 / (2 * paper_alpha)

        self.data_param = cp.Parameter(shape=meas_block.shape[0])
        self.nu = cp.Variable((space.nu_space.num_dirs, space.nu_space.total_points), "nu")
        data_term = self.data_scale_param * cp.sum_squares(meas_block @ cp.vec(self.nu.T) - self.data_param)
        if space.mu_space.num_dirs > 0:
            self.mu = cp.Variable((space.mu_space.num_dirs, space.mu_space.total_points), "mu")
            red_cons = cp.SOC(self.cons_bound_param, redcons_block @ cp.hstack((cp.vec(self.mu.T), cp.vec(self.nu.T))))
            tv_term = cp.norm(cp.vec(self.mu), 1) + cp.norm(cp.vec(self.nu), 1)
            # tv_term *= 3 / (space.mu_space.num_dirs * space.nu_space.num_dirs)
            # data_term = self.data_scale_param * cp.norm(meas_block @ cp.vec(self.nu.T) - data)
            obj = cp.Minimize(tv_term + data_term)
            self.prob = cp.Problem(obj, constraints=[red_cons])
        else:
            tv_term = cp.norm(cp.vec(self.nu), 1)
            obj = cp.Minimize(tv_term + data_term)
            self.prob = cp.Problem(obj)

    def set_target(self, target):
        self.data_param.value = target

    def solve(self, use_mosek=True):
        if use_mosek:
            self.prob.solve(solver=cp.MOSEK, verbose=True)
        else:
            self.prob.solve(verbose=True)

        if self.prob.status != "optimal":
            raise Exception("ERROR: cvxpy returned non-optimal result")

        if self.space.mu_space.num_dirs > 0:
            mu_res = self.space.mu_space.reshape(self.mu.value)
        else:
            mu_res = np.array([])

        return mu_res, self.space.nu_space.reshape(self.nu.value)
