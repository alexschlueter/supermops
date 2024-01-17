import numpy as np
import scipy.sparse as sp

from supermops.operators import MeasurementOperator, ReducedConsistencyOperator
from supermops.transforms import adapted_rad_range

from ..utils.utils import *
from .variables import LineProjectionSpace, MuSpace, NuSpace, SnapshotSpace


class ReducedLinearMotionSpace:
    """
    This class handles the grids used to discretize the dimension reduced
    model. It also provides methods to build the matrices which are used in the
    main objective and in the constraints.

    Variables:
    mu_dir - projection direction for joint Radon transform
    nu_dir - projection direction for Move operator reparametrized to S^1
    mu - joint Radon transform of phase space measure, one component per mu_dir
         each component is lives on a different 2D grid
    nu - reparametrized snapshot (result of reparametrized Move operator on
         phase space measure), one component per nu_dir

    For the optimization problem and the assembled matrices, the variables are
    packed into a flat vector in the order (mu, nu).
    The nu components corresponding to measurement times are packed at the end
    of the nu vector.
    """
    def __init__(
            self,
            main_bbox,
            times,
            num_rads,
            num_extra_nu_dirs,
            grid_size_mu,
            grid_sizes_nu,
            num_mu_dirs=None,
            mu_dirs=None,
            add_nu_dirs_for_times=True
    ):
        """
        Args:
            main_bbox: BoundingBox in which the particles stay at all times
            times: list of measurement times
            num_rads: number of bins for one-dimensional projections
            num_extra_nu_dirs: extra nu dirs to add to those coming from
                               measurement times (results in stronger time consistency)
            grid_size_mu: grid size along each dimension for variable mu
            grid_sizes_nu: list of grid sizes for nu, one for each dim
            num_mu_dirs: number of mu_dirs to generate automatically
            mu_dirs: list of 2D directions to use as mu_dirs
            add_nu_dirs_for_times: Whether to add nu_dirs corresponding to the
                                   measurement times
        """
        self.time_info = TimeInfo(times)
        self.main_bbox = main_bbox
        self.dim = main_bbox.dim()
        assert len(grid_sizes_nu) == self.dim
        assert isinstance(grid_size_mu, int)

        if mu_dirs is None:
            # TODO: generalize for dim != 2
            assert self.dim == 2
            assert num_mu_dirs is not None
            mu_dirs = np.array([
                (np.cos(angle), np.sin(angle)) for angle in np.linspace(
                    -np.pi / 2, np.pi / 2, num=num_mu_dirs, endpoint=False)
            ])

        self.mu_space = MuSpace(mu_dirs, grid_size_mu, main_bbox, self.time_info)

        self.num_rads = num_rads
        self.num_extra_nu_dirs = num_extra_nu_dirs

        num_total_nu_dirs = self.num_extra_nu_dirs
        if add_nu_dirs_for_times:
            num_total_nu_dirs += self.time_info.num_times

        nu_dirs = np.empty((num_total_nu_dirs, 2))
        if self.num_extra_nu_dirs > 0:
            nu_dirs[:self.num_extra_nu_dirs] = [
                (np.cos(angle), np.sin(angle))
                for angle in np.linspace(-np.pi / 2,
                                         np.pi / 2,
                                         num=self.num_extra_nu_dirs,
                                         endpoint=False)
            ]
        if add_nu_dirs_for_times:
            # add nu-dirs corresponding to times
            for ti, t in enumerate(self.time_info.times):
                cent_t = self.time_info.centered(t)
                nu_dirs[self.num_extra_nu_dirs + ti] = np.array((1, cent_t)) / np.sqrt(1 + cent_t**2)

        self.nu_space = NuSpace(nu_dirs, grid_sizes_nu, main_bbox, self.time_info)
        self.snapshot_space = SnapshotSpace(grid_sizes_nu, main_bbox, self.time_info)

        self.total_dofs = self.mu_space.total_dofs() + self.nu_space.total_dofs()

        self.rads = self.adapted_rad_grids()

    def adapted_rad_grids(self):
        """
        Generate bins for one-dimensional projections for each pair
        (mu_dir, nu_dir), each adapted to the expected range of projections from
        the grids of mu and nu
        """
        rads = np.empty(
            (self.mu_space.num_dirs, self.nu_space.num_dirs, self.num_rads))
        for mi, mu_dir in enumerate(self.mu_space.mu_dirs):
            for ni, nu_dir in enumerate(self.nu_space.nu_dirs):
                lo, hi = adapted_rad_range(mu_dir, nu_dir, self.main_bbox, self.time_info.tspan)
                rads[mi, ni] = np.linspace(lo, hi, self.num_rads)

        return rads

    def get_meas_time_entries(self, x):
        """
        Get only those entries from a flat solution vector corresponding to
        nu components for measurement times
        """
        return x[-self.time_info.num_times * self.nu_space.total_points:]

    def get_nu_for_time(self, nu, time_idx):
        """Get component of a nu array corresponding to a time index"""
        return nu[-self.time_info.num_times + time_idx]

    def unpack_sol_vec(self, sol):
        """Unpack a solution vector into variables mu, nu.

        Arguments:
            sol {ndarray} -- solution vector from optimization

        Returns:
            ndarray -- solution for mu for all mu dirs
            ndarray -- solution for nu for all nu dirs
        """
        if self.mu_space.num_dirs > 0:
            mu = self.mu_space.reshape(sol[:self.mu_space.total_dofs()])
        else:
            mu = None
        nu = self.nu_space.reshape(sol[self.mu_space.total_dofs():])

        return mu, nu

    def build_nu_meas_block(self, static_model):
        """
        Build a sparse matrix which calculates the measurement of a given
        static_model when applied to a nu vector

        Args:
            static_model: A ForwardModel to take measurements with
        """
        self.meas_op = MeasurementOperator(self.snapshot_space, static_model)
        self.meas_op.assemble()

        meas_block = sp.block_diag([self.meas_op.meas_mat] * self.time_info.num_times)
        zeros_bl = sp.coo_matrix(
            (meas_block.shape[0], self.nu_space.total_dofs() - meas_block.shape[1]))
        # prepend zero matrix for those entries of nu not corresponding to a
        # measurement (those from the extra nu_dirs)
        row = sp.hstack([zeros_bl, meas_block])
        return row

    def build_munu_meas_block(self, static_model):
        """Prepend zeroes to the meas block so that it can be applied to (mu,nu)"""
        meas_block = self.build_nu_meas_block(static_model)
        zeros_bl = sp.coo_matrix(
            (meas_block.shape[0], self.total_dofs - meas_block.shape[1]))
        row = sp.hstack([zeros_bl, meas_block])
        return row

    def build_redcons_block(self, *args, **kwargs):
        """
        Build a sparse matrix for the "projected / reduced time consistency"
        constraint, i.e. the linear map
        (mu, nu) -> Radon_{nu_dir} mu_{mu_dir} - Radon_{mu_dir} nu_{nu_dir}
        for each pair (mu_dir, nu_dir).
        """
        if self.mu_space.num_dirs == 0:
            return sp.csr_matrix((0, self.total_dofs))

        self.reduced_cons_op = ReducedConsistencyOperator(self, LineProjectionSpace())
        self.reduced_cons_op.assemble(*args, **kwargs)
        # nu_radon_mats, mu_radon_mats are 2d np.arrays of sparse matrices
        # since there is a different grid and thus a different radon matrix
        # for each nu[nu dir], mu[mu dir].
        # We need to have a block row for each combination of dirs
        mu_radon_block = sp.vstack([
            sp.block_diag(single_nu_all_mu_dirs)
            for single_nu_all_mu_dirs in self.reduced_cons_op.mu_radon_mats.transpose()
        ])
        nu_radon_block = -1 * sp.block_diag([
            sp.vstack(single_nu_all_mu_dirs)
            for single_nu_all_mu_dirs in self.reduced_cons_op.nu_radon_mats
        ])
        return sp.hstack([mu_radon_block, nu_radon_block])

    def build_main_objective(self, static_model, target, cons_scale=1.0):
        """
        Build full system matrix and right hand side vector for the
        dimension-reduced optimization problem. The system matrix consists of
        the block representing the measurement for each measurement time as well
        as the block for the reduced consistency constraint.

        Args:
            static_model: A ForwardModel to take measurements with
            target: measurement data
            cons_scale: Scale consistency constraint block. Defaults to 1.0.

        Returns:
            _description_
        """
        meas_block = self.build_meas_block(static_model)
        if self.mu_space.num_dirs == 0:
            Msys = meas_block
        else:
            redcons_block = self.build_redcons_block()
            redcons_block *= cons_scale
            Msys = sp.vstack([redcons_block, meas_block])
        y = np.hstack((np.zeros(Msys.shape[0] - target.shape[0]), target))

        nonzero = Msys.count_nonzero()
        percent_nz = nonzero / np.prod(Msys.shape) * 100
        print("Msys shape = {}, nonzero = {} ({:.3}%)".format(
            Msys.shape, nonzero, percent_nz))

        return Msys, y

    def dofs_per_var(self):
        return [c.total_dofs() for c in self.mu_space] + [c.total_dofs() for c in self.nu_space]

    def build_nonneg_sum_constr(self, l1_target):
        """Build matrix for nonnegativity constraint and sum <= l1_target
        constraint, as well as the corresponding right hand side vector.

        Arguments:
            l1_target {scalar} -- sum of each component variable should stay
                                  less than this number

        Returns:
            spmatrix -- sparse constraint matrix
            ndarray -- right hand side vector
        """
        nonneg_mat = -1 * sp.eye(self.total_dofs)
        dofs_per_var = self.dofs_per_var()
        sum_mat = sp.block_diag([dofs * [1] for dofs in dofs_per_var])
        constr_mat = sp.vstack([nonneg_mat, sum_mat])
        rhs = np.hstack((np.zeros(self.total_dofs),
                         np.full(len(dofs_per_var), l1_target)))

        return constr_mat, rhs

    def build_abs_constr(self):
        # need one additional var |x| for each dof x which represents absolute value of that dof
        # use constraints -|x| <= x <= |x|
        # block of new abs vars is ordered in front of real vars
        eye = sp.eye(self.total_dofs)
        return sp.bmat([
            [-1 * eye, -1 * eye],
            [-1 * eye, eye]
        ])

    def build_l1_constr(self, l1_target):
        """Build matrix for l1 (sum of absolute values) constraint
        and corresponding right hand side vector.

        Arguments:
            l1_target {scalar} -- l1 norm of each component variable should stay
                                  less than this number

        Returns:
            spmatrix -- sparse constraint matrix
            ndarray -- right hand side vector
        """
        abs_part = self.build_abs_constr()
        # sum up absolute values for each var
        dofs_per_var = self.dofs_per_var()
        sum_mat = sp.block_diag([dofs * [1.0] for dofs in dofs_per_var])
        zeros = sp.coo_matrix((sum_mat.shape[0], self.total_dofs))
        constr_mat = sp.vstack([abs_part,
            sp.hstack([
            sum_mat, zeros
        ])])  # zeros in bottom right because vars without abs are not summed

        rhs = np.hstack((
            np.zeros(2 * self.total_dofs),  # zeros for abs ineqs
            np.full(len(dofs_per_var), l1_target),
        ))  # sum to l1_target

        return constr_mat, rhs
