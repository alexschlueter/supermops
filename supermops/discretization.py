import itertools

import numpy as np
import scipy.sparse as sp

from supermops.utils import *


class Discretization:
    def __init__(
            self,
            K,
            dt,
            num_mu_dirs,
            num_rads,
            num_extra_nu_dirs,
            grid_size_mu,
            grid_size_nu,
            adaptive_grids=True,
    ):
        """This class handles the grids used to discretize the dimension reduced
        model. It also provides methods to build the linear systems which need
        to be solved for the main objective and the velocity objective.
        
        Arguments:
            K {int} -- time steps range from -K to K
            dt {float} -- time span between adjacent steps
            num_mu_dirs {int} -- number of directions for the variable mu
            num_rads {int} -- number of grid points to discretize projections
            num_extra_nu_dirs {int} -- number of directions for nu, excluding
                those already needed for each time step
            grid_size_mu {int} -- grid size along each dimension for variable mu
            grid_size_nu {int} -- grid size along each dimension for variable nu
        
        Keyword Arguments:
            adaptive_grids {bool} -- If True, a separate grid is generated for
                each mu dir, nu dir, exactly fitting the support of the measures
                for this direction. If False, oversized grids are generated
                which cover the supports for all directions. (default: {True})
        """
        self.K = K
        self.dt = dt
        self.num_times = 2 * K + 1
        self.times = np.linspace(-K, K, self.num_times)
        self.tmax = K * dt

        self.num_mu_dirs = num_mu_dirs
        self.num_rads = num_rads
        self.num_extra_nu_dirs = num_extra_nu_dirs

        self.grid_size_mu = grid_size_mu
        self.grid_size_nu = grid_size_nu
        self.adaptive_grids = adaptive_grids

        self.gs_nu_sq = self.grid_size_nu**2
        self.gs_mu_sq = self.grid_size_mu**2
        self.num_total_nu_dirs = self.num_extra_nu_dirs + self.num_times

        self.dofs_per_var = (self.num_mu_dirs * [self.gs_mu_sq] +
                             self.num_total_nu_dirs * [self.gs_nu_sq])
        self.total_dofs = np.sum(self.dofs_per_var)

        self.mu_dirs = np.array([
            (np.cos(angle), np.sin(angle)) for angle in np.linspace(
                -np.pi / 2, np.pi / 2, num=self.num_mu_dirs, endpoint=False)
        ])
        self.nu_dirs = np.empty((self.num_total_nu_dirs, 2))
        if self.num_extra_nu_dirs > 0:
            self.nu_dirs[:self.num_extra_nu_dirs] = [
                (np.cos(angle), np.sin(angle))
                for angle in np.linspace(-np.pi / 2,
                                         np.pi / 2,
                                         num=self.num_extra_nu_dirs,
                                         endpoint=False)
            ]
        # add nu-dirs corresponding to times
        for ti, t in enumerate(self.times):
            self.nu_dirs[self.num_extra_nu_dirs + ti] = np.array(
                (1, t)) / np.sqrt(1 + t**2)

        if self.adaptive_grids:
            self.rads = self.adapted_rad_grids()
            self.nu_grids = self.adapted_nu_grids()
            if self.num_mu_dirs > 0:
                self.mu_grids = self.adapted_mu_grids()
        else:
            self.mu_grid = self.oversized_mu_grid()
            self.nu_grid = self.oversized_nu_grid()
            self.rads = self.oversized_rad_grid()

    def oversized_mu_grid(self):
        """Generate one large grid which covers the supports of mu for all
        mu dirs.
        
        Returns:
            ndarray -- list of grid points
        """
        xgrid = np.linspace(-1, np.sqrt(2), self.grid_size_mu)
        ygrid = np.linspace(-1 / (np.sqrt(2) * self.tmax),
                            1 / (np.sqrt(2) * self.tmax), self.grid_size_mu)
        return np.array(list(itertools.product(xgrid, ygrid)))

    def oversized_nu_grid(self):
        """Generate one large grid which covers the supports of nu for all
        nu dirs.
        
        Returns:
            ndarray -- list of grid points
        """
        grid1 = np.linspace(
            -1 / (2 * self.tmax),
            np.maximum(1,
                       np.sqrt(1 + self.tmax**2) / (2 * self.tmax)),
            self.grid_size_nu)
        return np.array(list(itertools.product(grid1, grid1)))

    def oversized_rad_grid(self):
        """Generate one large 1d grid which covers projections of all points
        from the oversized grids.
        """
        lo, hi = [], []
        for grid, dirs in [(self.mu_grid, self.nu_dirs),
                           (self.nu_grid, self.mu_dirs)]:
            projs = (grid @ dirs.transpose()).flatten()
            lo.append(np.min(projs))
            hi.append(np.max(projs))

        return np.linspace(np.min(lo), np.max(hi), self.num_rads)

    def adapted_rad_grids(self):
        """For each pair of mu dir and nu dir, generate a grid which covers all
        projections of the grid points for mu[mu dir] onto the nu dir
        (or, symmetrically, projections of nu[nu dir] onto the mu dir).

        Assumes that x-coords of all dirs are positive.
        Uses exact expressions for min / max radii. Could alternatively
        calculate these numerically from grids + dirs.
        
        Returns:
            ndarray -- list of grid points for each mu dir, nu dir
        """
        rads = np.empty(
            (self.num_mu_dirs, self.num_total_nu_dirs, self.num_rads))
        for mi, m in enumerate(self.mu_dirs):
            for ni, n in enumerate(self.nu_dirs):
                vscal = abs(n[1]) / self.tmax
                if n[0] - vscal >= 0:
                    if m[1] < 0:
                        left, right = (m[1] * n[0], m[0] * n[0])
                    else:
                        left, right = (0, np.sum(m) * n[0])
                else:
                    minus, plus = 0.5 * (n[0] - vscal), 0.5 * (n[0] + vscal)
                    if m[1] < 0:
                        left, right = (
                            m[0] * minus + m[1] * plus,
                            m[0] * plus + m[1] * minus,
                        )
                    else:
                        left, right = (
                            m[0] * minus + m[1] * minus,
                            m[0] * plus + m[1] * plus,
                        )

                rads[mi, ni] = np.linspace(left, right, self.num_rads)

        return rads

    def adapted_mu_grids(self):
        """For each mu dir, generate a grid which covers the support of
        mu[mu dir].

        Returns:
            ndarray -- grid points for each mu dir
        """
        std_grid1 = np.linspace(0, 1, self.grid_size_mu)
        std_grid = np.array(list(itertools.product(std_grid1, std_grid1)))
        rot45 = 0.5 * np.array([[1, 1], [-1 / self.tmax, 1 / self.tmax]])
        rot_grid = std_grid @ rot45.transpose()
        res = np.empty((self.num_mu_dirs, len(std_grid), 2))
        for di, d in enumerate(self.mu_dirs):
            # assuming that we always have d[0] >= 0
            if d[1] > 0:
                res[di] = rot_grid * np.sum(d)
            else:
                res[di] = rot_grid * (d[0] - d[1]) + (d[1], 0)

        return res

    def adapted_nu_grid_1d(self, dir):
        """Generate 1D grid for nu[dir]

        Assumes dir[0] >= 0.
        
        Arguments:
            dir {ndarray} -- nu dir to generate grid for
        
        Returns:
            ndarray -- grid points
        """
        vscal = abs(dir[1]) / self.tmax
        if dir[0] - vscal >= 0:
            grid_nu1 = np.linspace(0, dir[0], self.grid_size_nu)
        else:
            grid_nu1 = np.linspace(0.5 * (dir[0] - vscal),
                                   0.5 * (dir[0] + vscal), self.grid_size_nu)
        return grid_nu1

    def adapted_nu_grids(self):
        """For each nu dir, generate a grid which covers the support of
        nu[nu dir]
        
        Returns:
            ndarray -- grid points for each nu dir
        """
        res = np.empty((self.num_total_nu_dirs, self.gs_nu_sq, 2))
        for ni, n in enumerate(self.nu_dirs):
            grid_nu1 = self.adapted_nu_grid_1d(n)
            res[ni] = np.array(list(itertools.product(grid_nu1, grid_nu1)))

        return res

    def ass_sparse_sys_matrix(self, meas_mats, nu_radon_mats, mu_radon_mats):
        """Assemble the main system matrix which contains the matrices for all
        projection terms as well as the measurement matrices.
        
        Arguments:
            meas_mats {ndarray} -- measurement matrices for each time step
            nu_radon_mats {ndarray} -- sparse projection matrices for nu for
                each pair of nu dir, mu dir
            mu_radon_mats {ndarray]} -- sparse projection matrices for mu for
                each pair of mu dir, nu dir
        
        Returns:
            spmatrix -- sparse system matrix
        """
        meas_block = sp.block_diag(meas_mats)

        if self.num_mu_dirs == 0:
            # no projections => no coupling between time steps
            # => static reconstructions
            return meas_block

        if self.adaptive_grids:
            # nu_radon_mats, mu_radon_mats are 2d np.arrays of sparse matrices
            # since there is a different grid and thus a different radon matrix
            # for each nu[nu dir], mu[mu dir].
            # We need to have a block row for each combination of dirs
            nu_radon_block = -1 * sp.block_diag([
                sp.vstack(single_nu_all_mu_dirs)
                for single_nu_all_mu_dirs in nu_radon_mats
            ])
            mu_radon_block = sp.vstack([
                sp.block_diag(single_nu_all_mu_dirs)
                for single_nu_all_mu_dirs in mu_radon_mats.transpose()
            ])
        else:
            # nu_radon_mats, mu_radon_mats are lists of sparse matrices
            nu_radon_block = -1 * sp.block_diag(
                self.num_total_nu_dirs * [sp.vstack(nu_radon_mats)])
            mu_radon_block = sp.vstack([
                sp.block_diag(self.num_mu_dirs * [single_nu_dir])
                for single_nu_dir in mu_radon_mats
            ])

        row0 = sp.hstack([mu_radon_block, nu_radon_block])
        zeros_bl = sp.coo_matrix(
            (meas_block.shape[0], row0.shape[1] - meas_block.shape[1]))
        row1 = sp.hstack([zeros_bl, meas_block])
        Msys = sp.vstack([row0, row1])

        return Msys

    def build_main_objective(self, model, target):
        """Build linear system representing the constraints in the dimension
        reduced problem. These consist of the measurement consistency constraints
        for each time step as well as the projection conditions for variables
        mu and nu.
        
        Arguments:
            model {ForwardModel} -- the measurement model
            target {ndarray} -- target measurement
        
        Returns:
            spmatrix -- sparse system matrix
            ndarray -- vector for right hand side
        """
        assert np.all(
            model.get_param_bounds() == [[0, 0], [1, 1]]
        ), "Only 2D models with bounds [0,1]^2 supported for now!"

        if self.adaptive_grids:
            # build matrices for line projections of mu
            # for each mu dir, mu[mu dir] is projected onto the direction of
            # each nu dir
            mu_radon_mats = np.empty(
                (self.num_mu_dirs, self.num_total_nu_dirs), dtype=object)
            for di in range(self.num_mu_dirs):
                print("Building mu radon mats: {}/{}".format(
                    di + 1, self.num_mu_dirs))
                mu_radon_mats[di] = build_radon_matrix(self.mu_grids[di],
                                                       self.nu_dirs,
                                                       self.rads[di])

            # build matrices for line projections of nu
            # for each nu dir, nu[nu dir] is projected onto the direction of
            # each mu dir
            nu_radon_mats = np.empty(
                (self.num_total_nu_dirs, self.num_mu_dirs), dtype=object)
            if self.num_mu_dirs > 0:
                for di in range(self.num_total_nu_dirs):
                    print("Building nu radon mats: {}/{}".format(
                        di + 1, self.num_total_nu_dirs))
                    nu_radon_mats[di] = build_radon_matrix(
                        self.nu_grids[di], self.mu_dirs, self.rads[:, di])

            # build measurement matrix for each time step by applying forward
            # model to each grid point
            # the grid points for nu need to be scaled since
            # u_k = [x -> sqrt(1+(k*dt)^2)*x]_# nu(n_k, .)
            meas_mats = np.empty(
                (self.num_times, model.meas_size(), self.gs_nu_sq))
            for ti, t in enumerate(self.times):
                nu_grid_for_time = self.nu_grids[-self.num_times + ti]
                for j, pnt in enumerate(nu_grid_for_time):
                    tpnt = pnt * np.sqrt(1 + t**2)
                    meas_mats[ti, :, j] = model.apply_single_source(tpnt)

        else:  # using oversized grids
            print("Building radon matrices for mu...")
            mu_radon_mats = build_radon_matrix(self.mu_grid, self.nu_dirs,
                                               self.rads)
            print("Building radon matrices for nu...")
            nu_radon_mats = build_radon_matrix(self.nu_grid, self.mu_dirs,
                                               self.rads)

            print("Building measurement matrices...")
            meas_mats = np.zeros(
                (self.num_times, model.meas_size(), self.gs_nu_sq))
            for ti, t in enumerate(self.times):
                for j, pnt in enumerate(self.nu_grid):
                    tpnt = pnt * np.sqrt(1 + t**2)
                    # True sources should lie inside domain [0,1]^2 at all times,
                    # but grid points from oversized grid may land outside.
                    # If we set the measurement matrix to zero the variables
                    # corresponding to these points should also become zero
                    # due to the l1 regularization later.
                    if np.all(0 <= tpnt) and np.all(tpnt <= 1):
                        meas_mats[ti, :, j] = model.apply_single_source(tpnt)

        print("Building Msys...")
        Msys = self.ass_sparse_sys_matrix(meas_mats, nu_radon_mats,
                                          mu_radon_mats)
        y = np.hstack((np.zeros(Msys.shape[0] - target.shape[0]), target))

        nonzero = Msys.count_nonzero()
        percent_nz = nonzero / np.prod(Msys.shape) * 100
        print("Msys shape = {}, nonzero = {} ({:.3}%)".format(
            Msys.shape, nonzero, percent_nz))

        return Msys, y

    def build_nonneg_sum_constr(self, l1_target):
        """Build matrix for nonnegativity constraint and sum <= l1_target
        constraint, as well as the corresponding right hand side vector.
        
        Arguments:
            l1_target {scalar} -- sum should stay less than this number
        
        Returns:
            spmatrix -- sparse constraint matrix
            ndarray -- right hand side vector
        """
        nonneg_mat = -1 * sp.eye(self.total_dofs)
        sum_mat = sp.block_diag([dofs * [1] for dofs in self.dofs_per_var])
        constr_mat = sp.vstack([nonneg_mat, sum_mat])
        rhs = np.hstack((np.zeros(self.total_dofs),
                         np.full(len(self.dofs_per_var), l1_target)))

        return constr_mat, rhs

    def build_l1_constr(self, l1_target):
        """Build matrix for l1 (sum of absolute values) constraint
        and corresponding right hand side vector.
        
        Arguments:
            l1_target {scalar} -- l1 norm should stay less than this number
        
        Returns:
            spmatrix -- sparse constraint matrix
            ndarray -- right hand side vector
        """
        # need one additional var |x| for each dof x which represents absolute value of that dof
        # use constraints -|x| <= x <= |x|
        # block of new abs vars is ordered in front of real vars
        data = np.full(2 * self.total_dofs, -1.0)
        I = np.arange(2 * self.total_dofs)
        J = np.repeat(np.arange(self.total_dofs), 2)
        abs_part = sp.coo_matrix((data, (I, J)))

        data = np.tile([1.0, -1.0], self.total_dofs)
        I = np.arange(2 * self.total_dofs)
        J = np.repeat(np.arange(self.total_dofs), 2)
        dof_part = sp.coo_matrix((data, (I, J)))

        # sum up absolute values for each var
        sum_mat = sp.block_diag([dofs * [1.0] for dofs in self.dofs_per_var])
        constr_mat = sp.bmat([[abs_part, dof_part], [
            sum_mat, None
        ]])  # zeros in bottom right because vars without abs are not summed

        rhs = np.hstack((
            np.zeros(2 * self.total_dofs),  # zeros for abs ineqs
            np.full(len(self.dofs_per_var), l1_target),
        ))  # sum to l1_target

        return constr_mat, rhs

    def build_vel_objective(self, mu):
        """Build matrix and right hand side for the velocity problem, which
        tries to reconstruct the velocities of the particles from a solution
        mu of the main objective.
        
        Arguments:
            mu {ndarray} -- solution for mu from main objective
        
        Returns:
            spmatrix -- velocity matrix
            ndarray -- right hand side vector
        """
        assert self.num_mu_dirs > 0, "Cannot build velocity problem with num_mu_dirs = 0"

        std_grid1 = np.linspace(0, 1, self.grid_size_nu)
        std_grid = np.array(list(itertools.product(std_grid1, std_grid1)))

        if self.adaptive_grids:
            V_radon_mats = build_radon_matrix(std_grid, self.mu_dirs,
                                              self.rads[:, -self.K - 1])
        else:
            V_radon_mats = build_radon_matrix(std_grid, self.mu_dirs,
                                              self.rads)

        data = []
        row = []
        col = []
        for di, d in enumerate(self.mu_dirs):
            coo_radon_for_dir = V_radon_mats[di].tocoo(copy=False)
            data += list(d[0] * coo_radon_for_dir.data)
            row += list((di * self.num_rads) + coo_radon_for_dir.row)
            col += list(coo_radon_for_dir.col)

            data += list(d[1] * coo_radon_for_dir.data)
            row += list((di * self.num_rads) + coo_radon_for_dir.row)
            col += list(len(std_grid) + coo_radon_for_dir.col)

        Vmat = sp.coo_matrix(
            (data, (row, col)),
            shape=(self.num_mu_dirs * self.num_rads, 2 * len(std_grid)))

        if self.adaptive_grids:
            mu_radons = []
            for di, d in enumerate(self.mu_dirs):
                mu_radons.append(
                    build_radon_matrix(self.mu_grids[di], [[1, 0]],
                                       self.rads[np.newaxis, di, -self.K -
                                                 1])[0])

            mumat = sp.block_diag([
                mu_radons[di].multiply(self.mu_grids[di, :, 1])
                for di in range(self.num_mu_dirs)
            ])
        else:
            mu_radon = build_radon_matrix(self.mu_grid, [[1, 0]], self.rads)[0]
            mumat = sp.block_diag(self.num_mu_dirs *
                                  [mu_radon.multiply(self.mu_grid[:, 1])])

        b = mumat @ mu.flatten()

        return Vmat, b

    def unpack_sol_vec(self, sol):
        """Unpack a solution vector into variables mu, nu.
        
        Arguments:
            sol {ndarray} -- solution vector from optimization
        
        Returns:
            ndarray -- solution for mu for all mu dirs
            ndarray -- solution for nu for all nu dirs
        """
        if self.num_mu_dirs > 0:
            mu = sol[:self.num_mu_dirs * self.gs_mu_sq].reshape(
                self.num_mu_dirs, self.grid_size_mu, -1)
        else:
            mu = None
        nu = sol[self.num_mu_dirs * self.gs_mu_sq:].reshape(
            self.num_total_nu_dirs, self.grid_size_nu, -1)
        return mu, nu

    def nu_sol_for_time(self, nu, k):
        return nu[-self.K - 1 + k]
