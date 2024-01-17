import itertools
import math
from multiprocessing import Pool

import numpy as np
import psutil
import scipy.sparse as sp

from supermops.geometry import BoundingBox
from supermops.transforms import (calc_mu_params, normalized_jrad_coords,
                                normalized_move_coords)
from supermops.utils.utils import profile

from .projectors import center_point_projector, strip_projector_2d


class MeasurementOperator:
    """
    This class manages the assembled matrix for a measurement model applied to a
    variable on the grid from snapshot_space.

    Note that we can directly use this matrix for the values in the nu vector as
    well, even though the nu grid is an affinely transformed version of the
    snapshot grid. This is possible as long as the ordering between the nu grids
    and the snapshot grid is consistent, because we only take the values of nu,
    not the grid positions.
    """

    def __init__(self, snapshot_space, static_model):
        self.snapshot_space = snapshot_space
        self.static_model = static_model

    def assemble(self):
        snap_grid = self.snapshot_space.get_center_grid()
        self.meas_mat = self.static_model.apply_multiple(snap_grid)

    def __call__(self, snaps):
        res = np.empty(len(snaps) * self.meas_mat.shape[0])
        for snap, rows in zip(snaps, res.reshape(len(snaps), -1)):
            rows[:] = self.meas_mat @ snap.flatten()
        return res

class ReducedConsistencyOperator:
    """
    This class manages the assembled matrices for the dimension-reduced time
    consistency constraint between mu und nu, i.e. the operations
    Radon_{nu_dir} mu_{mu_dir} for all mu_dir, nu_dir and
    Radon_{mu_dir} nu_{nu_dir} for all mu_dir, nu_dir.
    """

    def __init__(self, domain, codomain):
        """
        Args:
            domain: ReducedLinearMotionSpace
            codomain: LineProjectionSpace, currently not used
        """
        self.domain = domain
        self.codomain = codomain

    @staticmethod
    def do_project_new(comp, proj_dir, rads):
        return strip_projector_2d(comp, proj_dir, rads)

    @staticmethod
    def do_project_old(comp, proj_dir, rads):
        return center_point_projector(comp.get_center_grid(), [proj_dir], rads)

    def assemble(self, projector="new", procs=len(psutil.Process().cpu_affinity())):
        """Assemble the matrices for the Radon operator for each pair of mu_dir, nu_dir

        Projection methods:
        new: Define bins on the projection direction, then for each bin, cover
             the 2D grid by the strip perpendicular to the proj. dir with
             boundaries defined by the bin boundaries. The area of intersection
             of each grid cell with this strip determines the contribution of
             the value at this grid cell to the value in the 1D bin.
        old: Project the center point of each grid cell onto the projection dir,
             perform linear interpolation to determine the contribution to the
             closest two discretization points on the line.

        Args:
            projector: Use "new" or "old" projection method. Defaults to "new".
            procs: Number of processes for parallel assembly.
                   Defaults to number of usable CPUs.
        """
        if projector == "new":
            do_project = self.do_project_new
            def do_unpack(res):
                return res
        elif projector == "old":
            do_project = self.do_project_old
            def do_unpack(res):
                return res[0]
        else:
            raise

        with Pool(procs) as pool:
            self.mu_radon_mats = np.empty(
                (self.domain.mu_space.num_dirs, self.domain.nu_space.num_dirs), dtype=object)
            for mi, mu_comp in enumerate(self.domain.mu_space):
                for ni, nu_comp in enumerate(self.domain.nu_space):
                    self.mu_radon_mats[mi, ni] = pool.apply_async(do_project, (mu_comp, nu_comp.nu_dir, self.domain.rads[mi, ni]))

            self.nu_radon_mats = np.empty(
                (self.domain.nu_space.num_dirs, self.domain.mu_space.num_dirs), dtype=object)
            # if self.domain.mu_space.num_dirs > 0:
            for ni, nu_comp in enumerate(self.domain.nu_space):
                for mi, mu_comp in enumerate(self.domain.mu_space):
                    self.nu_radon_mats[ni, mi] = pool.apply_async(do_project, (nu_comp, mu_comp.mu_dir, self.domain.rads[mi, ni]))

            for mi, mu_comp in enumerate(self.domain.mu_space):
                print(f"Building mu radon mats: {mi+1}/{self.domain.mu_space.num_dirs}")
                for ni, nu_comp in enumerate(self.domain.nu_space):
                    self.mu_radon_mats[mi, ni] = do_unpack(self.mu_radon_mats[mi, ni].get())

            for ni, nu_comp in enumerate(self.domain.nu_space):
                print(f"Building nu radon mats: {ni+1}/{self.domain.nu_space.num_dirs}")
                for mi, mu_comp in enumerate(self.domain.mu_space):
                    self.nu_radon_mats[ni, mi] = do_unpack(self.nu_radon_mats[ni, mi].get())

        # scale = 1 / (self.domain.mu_space.num_dirs * self.domain.nu_space.num_dirs)
        # self.mu_radon_mats *= scale
        # self.nu_radon_mats *= scale

    def __call__(self, mu, nu, out=None):
        if out is None:
            res = np.empty((self.domain.mu_space.num_dirs, self.domain.nu_space.num_dirs, self.mu_radon_mats[0,0].shape[0]))
        else:
            res = out

        for mi in range(self.domain.mu_space.num_dirs):
            for ni in range(self.domain.nu_space.num_dirs):
                res[mi, ni] = self.nu_radon_mats[ni,mi] @ nu[ni].flatten()
                res[mi, ni] -= self.mu_radon_mats[mi,ni] @ mu[mi].flatten()

        return res

    def adjoint(self, y, out=None):
        if out is None:
            mu = np.empty((self.domain.mu_space.num_dirs, self.domain.mu_space.total_points))
            nu = np.empty((self.domain.nu_space.num_dirs, self.domain.nu_space.total_points))
        else:
            mu, nu = out

        for mi in range(self.domain.mu_space.num_dirs):
            for ni in range(self.domain.nu_space.num_dirs):
                mu[mi] -= self.mu_radon_mats[mi, ni].transpose() @ y[mi, ni]
                nu[ni] += self.nu_radon_mats[ni, mi].transpose() @ y[mi, ni]

        return mu, nu

class JointRadonOperator:
    """
    This class manages the assembled matrices for the joint Radon operator
    applied to the phase space measure, i.e. the operation
    jointRadon_{mu_dir} lambda for each mu_dir

    CAUTION: Implementation not well tested
    """

    def __init__(self, domain, codomain):
        """
        Args:
            domain: PhaseSpace
            codomain: MuSpace
        """
        self.domain = domain
        self.codomain = codomain

    @staticmethod
    @profile
    def assemble_one_dir(domain, mu_space, permut_mat):
        mu_cell_area = mu_space[0,0].area()
        phase_jrad_centers = normalized_jrad_coords(domain.get_std_1d_grids(), mu_space.mu_dir, domain.main_bbox)
        _, spos, sneg = mu_space.calc_mu_params()
        s = spos - sneg
        _, mspos, msneg = calc_mu_params(mu_space.mu_dir, BoundingBox(domain.main_bbox.bounds / domain.grid_size_per_dim[:,None]))
        ms = mspos - msneg
        side_len = ms / s
        area_factor = domain.cell_volume() / (side_len ** 2 * mu_cell_area * mu_space.total_points)
        # sort_idc = np.argsort(phase_jrad_centers)
        # sorted_points = phase_jrad_centers[sort_idc]
        left = phase_jrad_centers - side_len / 2
        right = phase_jrad_centers + side_len / 2
        left_idc = np.floor(mu_space.grid_size * left).astype(int)
        right_idc = np.ceil(mu_space.grid_size * right).astype(int) - 1
        left_overlaps = (left_idc + 1) - left * mu_space.grid_size
        right_overlaps = right * mu_space.grid_size - right_idc

        filled_1d = []
        for left_idx, right_idx, left_overlap, right_overlap in zip(left_idc, right_idc, left_overlaps, right_overlaps):
            if left_idx == right_idx:
                values = [side_len * mu_space.grid_size]
            else:
                # values = [left_overlap] + [mu_cell_area] * (right_idx - left_idx - 1) + [right_overlap]
                values = [left_overlap] + [1.0] * (right_idx - left_idx - 1) + [right_overlap]
            filled_1d.append((range(left_idx, right_idx + 1), values))

        rows, cols, data = [], [], []
        for col, ((r1, v1), (r2, v2)) in enumerate(itertools.product(filled_1d, filled_1d)):
            filled_2d = np.array(list(itertools.product(r1, r2)))
            rows += list(np.ravel_multi_index(filled_2d.transpose(), mu_space.grid_sizes(), mode="clip"))
            cols += len(filled_2d) * [col]
            # print("debug jrad", mu_space.mu_dir, ms, s, side_len, area_factor, np.outer(v1, v2).flatten(), filled_1d, left_overlap, right_overlap, left, right, left_idc, right_idc)
            # data += list(area_factor * np.outer(v1, v2).flatten())
            data += [area_factor * vv1 * vv2 for vv1 in v1 for vv2 in v2]
            # print([area_factor * vv1 * vv2 for vv1 in v1 for vv2 in v2], list(area_factor * np.outer(v1, v2).flatten()))
            # a = [area_factor * vv1 * vv2 for vv1 in v1 for vv2 in v2]
            # b = [area_factor * vv1 * vv2 for vv2 in v2 for vv1 in v1]
            # c =
            # if a != b:
            # print(np.linalg.norm([area_factor * vv1 * vv2 for vv1 in v1 for vv2 in v2] - area_factor * np.outer(v1, v2).flatten()))
            # assert [area_factor * vv1 * vv2 for vv2 in v2 for vv1 in v1] == list(area_factor * np.outer(v1, v2).flatten())

        return sp.coo_matrix((data, (rows, cols)), shape=(mu_space.total_points, domain.total_points)) @ permut_mat

    def assemble(self):
        col_reorder = np.arange(self.domain.total_points).reshape(self.domain.grid_sizes())
        col_reorder = col_reorder.transpose(
            list(itertools.chain(range(0, 2*self.domain.dim, 2), range(1, 2*self.domain.dim, 2)))
        )

        permut_mat = sp.eye(self.domain.total_points).tocoo()
        permut_mat.col = col_reorder.flatten()

        self.matrices = []
        with Pool(len(psutil.Process().cpu_affinity())) as pool:
            procs = []
            for mu_idx, mu_space in enumerate(self.codomain):
                procs.append(pool.apply_async(self.assemble_one_dir, (self.domain, mu_space, permut_mat)))
            for mu_idx, proc in enumerate(procs):
                print(f"Assembling joint radon matrix for mu dir {mu_idx+1}/{self.codomain.num_components}")
                self.matrices.append(proc.get())
        # for mu_idx, mu_space in enumerate(self.codomain.get_mu_components()):
        #     print(f"Assembling joint radon matrix for mu dir {mu_idx+1}/{self.codomain.num_mu_dirs}")
        #     self.matrices.append(self.assemble_one_dir(self.domain, mu_space))

    def __call__(self, phase_var, out=None):
        if out is None:
            res = np.empty((self.codomain.num_components, self.codomain.total_points))
        else:
            res = out

        # phase_var = phase_var.reshape(self.domain.grid_sizes())
        # phase_var = phase_var.transpose(
        #     list(itertools.chain(range(0, 2*self.domain.dim, 2), range(1, 2*self.domain.dim, 2)))
        # )
        # phase_var = phase_var.reshape(np.prod(self.domain.grid_sizes), -1)
        phase_var = phase_var.flatten()

        for di in range(self.codomain.num_components):
            res[di] = self.matrices[di] @ phase_var

        return res

    def adjoint(self, mu, out=None):
        if out is None:
            res = np.empty(self.domain.total_points)
        else:
            assert False
            res = out # TODO: this ain't gonna work with all the reshaping

        for di in range(self.codomain.num_components):
            # print(self.matrices[di].shape, mu[di].shape)
            res += self.matrices[di].transpose() @ mu[di].flatten()

        # res = res.reshape(np.tile(self.domain.grid_size_per_dim, 2))
        # res = res.transpose(
        #     list(itertools.chain(range(0, 2*self.domain.dim, 2), range(1, 2*self.domain.dim, 2)))
        # )

        return res.reshape(self.domain.points_per_dim)

class MoveOperator:
    """
    This class manages the assembled matrices for the reparametrized Move
    operator applied to the phase space measure, i.e. the operation
    rMove_{nu_dir} lambda for each nu_dir

    CAUTION: Implementation not well tested
    """

    def __init__(self, domain, codomain):
        """
        Args:
            domain: PhaseSpace
            codomain: NuSpace
        """
        self.domain = domain
        self.codomain = codomain

    @staticmethod
    @profile
    def assemble_one_dir(domain, nu_space):
        nu_cell_area = nu_space[0,0].area()
        phase_move_centers = normalized_move_coords(domain.get_grids(), nu_space.nu_dir, domain.main_bbox, domain.time_info.tspan)
        side_lens = 1 / domain.grid_size_per_dim
        area_factor = domain.cell_volume() / (np.prod(side_lens) * nu_cell_area * nu_space.total_points)
        # sort_idc = np.argsort(phase_jrad_centers)
        # sorted_points = phase_jrad_centers[sort_idc]
        left_idc, right_idc, left_overlaps, right_overlaps = [], [], [], []
        for cents_for_dim, side_len, grid_size in zip(phase_move_centers, side_lens, nu_space.grid_sizes()):
            left = cents_for_dim - side_len / 2
            right = cents_for_dim + side_len / 2
            left_idc_for_dim = np.floor(grid_size * left).astype(int)
            left_idc.append(left_idc_for_dim)
            right_idc_for_dim = np.ceil(grid_size * right).astype(int) - 1
            right_idc.append(right_idc_for_dim)
            left_overlaps.append((left_idc_for_dim + 1) - left * grid_size)
            right_overlaps.append(right * grid_size - right_idc_for_dim)

        filled_1d = []
        for left_idc_for_dim, right_idc_for_dim, left_overlaps_for_dim, right_overlaps_for_dim, side_len, grid_size in zip(left_idc, right_idc, left_overlaps, right_overlaps, side_lens, nu_space.grid_sizes()):
            filled_1d_for_dim = []
            for left_idx, right_idx, left_overlap, right_overlap in zip(left_idc_for_dim, right_idc_for_dim, left_overlaps_for_dim, right_overlaps_for_dim):
                if left_idx == right_idx:
                    values = [side_len * grid_size]
                else:
                    values = [left_overlap] + [1.0] * (right_idx - left_idx - 1) + [right_overlap]
                filled_1d_for_dim.append((range(left_idx, right_idx + 1), values))

            filled_1d.append(filled_1d_for_dim)

        rows, cols, data = [], [], []
        for col, filled_comb in enumerate(itertools.product(*filled_1d)):
            # print("pre", len(rows), len(cols), len(data))
            filled_full = np.array(list(itertools.product(*(idc for idc, _ in filled_comb))))
            # print([idc for idc, _ in filled_comb])
            # print(filled_full)
            rows += list(np.ravel_multi_index(filled_full.transpose(), nu_space.grid_sizes(), mode="clip"))
            cols += len(filled_full) * [col]
            # print([vals for _, vals in filled_comb])
            # print(np.outer(*(vals for _, vals in filled_comb)))
            # print(list(np.outer(*(vals for _, vals in filled_comb)).flatten()))
            # data += list(area_factor * np.outer(*(vals for _, vals in filled_comb)).flatten()) # TODO: multidim outer?
            data += [area_factor * math.prod(values) for values in itertools.product(*(vals for _, vals in filled_comb))] # TODO: multidim outer?
            # print(len(rows), len(cols), len(data))

        return sp.coo_matrix((data, (rows, cols)), shape=(nu_space.total_points, domain.total_points))

    def assemble(self):
        self.matrices = []
        with Pool(len(psutil.Process().cpu_affinity())) as pool:
            procs = []
            for nu_idx, nu_space in enumerate(self.codomain):
                procs.append(pool.apply_async(self.assemble_one_dir, (self.domain, nu_space)))
            for nu_idx, proc in enumerate(procs):
                print(f"Assembling move matrix for nu dir {nu_idx+1}/{self.codomain.num_components}")
                self.matrices.append(proc.get())
        # for nu_idx, nu_space in enumerate(self.codomain.get_nu_components()):
        #     print(f"Assembling move matrix for nu dir {nu_idx+1}/{self.codomain.num_total_nu_dirs}")
        #     self.matrices.append(self.assemble_one_dir(self.domain, nu_space))

    def __call__(self, phase_var, out=None):
        if out is None:
            res = np.empty((self.codomain.num_dirs, self.codomain.total_points))
        else:
            res = out

        # phase_var = phase_var.reshape(np.repeat(self.domain.grid_sizes, 2))
        # phase_var = phase_var.transpose(
        #     list(itertools.chain(range(0, 2*self.domain.dim, 2), range(1, 2*self.domain.dim, 2)))
        # )
        # phase_var = phase_var.reshape(np.prod(self.domain.grid_sizes), -1)
        phase_var = phase_var.flatten()

        for di in range(self.codomain.num_components):
            res[di] = self.matrices[di] @ phase_var

        return res

    def adjoint(self, nu, out=None):
        if out is None:
            res = np.empty(self.domain.total_points)
        else:
            res = out

        for di in range(self.codomain.num_components):
            res += self.matrices[di].transpose() @ nu[di].flatten()

        return res.reshape(self.domain.points_per_dim)