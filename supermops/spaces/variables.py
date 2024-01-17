import itertools
import math

from matplotlib import transforms as mpltransforms
from supermops.geometry import BoundingBox, Parallelogram

from supermops.transforms import *
from supermops.utils.utils import TimeInfo, profile

from .abstract import FiniteElementSpace, RegularProductSpace, ShapedSpace
from .strippable import StrippableSpace


class MuComponentSpace(ShapedSpace, FiniteElementSpace, StrippableSpace):
    """
    Represents a component of mu for a given mu_dir
    """

    def __init__(self, mu_dir, grid_size, main_bbox, time_info):
        ShapedSpace.__init__(self, (grid_size, grid_size))
        self.mu_dir = mu_dir
        self.grid_size = grid_size
        self.total_points = grid_size ** 2
        self.main_bbox = main_bbox
        self.time_info = time_info

        self.tr_mat, self.tr_offs = get_mu_transform(self.mu_dir, self.main_bbox, self.time_info.tspan)
        # self.grid = self.adapted_mu_grid()

    def grid_sizes(self):
        return self.grid_size, self.grid_size

    def calc_cell_area(self):
        """Area of a single grid cell"""
        _, spos, sneg = calc_mu_params(self.mu_dir, self.main_bbox)
        s = spos - sneg
        return s ** 2 / (self.time_info.tspan * self.total_points)

    def calc_mu_params(self):
        return calc_mu_params(self.mu_dir, self.main_bbox)

    def get_transform(self):
        # return get_mu_transform(self.mu_dir, self.main_bbox, self.time_info.tspan)
        return self.tr_mat, self.tr_offs

    def get_inverse_transform(self):
        return get_inverse_mu_transform(self.mu_dir, self.main_bbox, self.time_info.tspan)

    def inverse_mu_transform(self, points):
        return inverse_mu_transform(points, self.mu_dir, self.main_bbox, self.time_info.tspan)

    def adapted_mu_grid(self):
        """
        Generate a grid covering the support of mu for this mu_dir.
        Returns a list of 2D grid cell centers.
        """
        grid0 = np.linspace(0, 1, self.grid_size + 1)
        grid0 = (grid0[1:] + grid0[:-1]) / 2
        mu_unit_grid = np.array(list(itertools.product(grid0, grid0)))
        trmat, offset = self.get_transform()
        res = mu_unit_grid @ trmat.transpose()
        res += offset
        return res

    @profile
    def __getitem__(self, idx):
        """Return the Parallelogram object for the grid cell at a given index"""

        # std_box = BoundingBox([[0, 1], [0, 1]]).get_grid_cell((self.grid_size, self.grid_size), idx)
        # corners = std_box.corners()
        x0, y0 = idx[0] / self.grid_size, idx[1] / self.grid_size
        step = 1 / self.grid_size
        corners = [[x0, y0 + step], [x0, y0], [x0 + step, y0]]
        trmat, offset = self.get_transform()
        tr_corners = corners @ trmat.transpose()
        tr_corners += offset
        return Parallelogram(*tr_corners)

class NuComponentSpace(ShapedSpace, FiniteElementSpace, StrippableSpace):
    """
    Represents a component of nu for a given nu_dir
    """

    def __init__(self, nu_dir, grid_sizes, main_bbox, time_info):
        ShapedSpace.__init__(self, grid_sizes)
        self.nu_dir = nu_dir
        self._grid_sizes = np.asarray(grid_sizes)
        self.total_points = np.prod(grid_sizes)
        self.main_bbox = main_bbox
        self.time_info = time_info

        self.nu_bbox = self.get_nu_bbox()

    def grid_sizes(self):
        return self._grid_sizes

    def nu_std_bounds(self):
        return nu_std_bounds(self.nu_dir, self.time_info.tspan)

    def get_nu_bbox(self):
        return get_nu_bbox(self.nu_dir, self.main_bbox, self.time_info.tspan)

    def get_transform(self):
        return get_nu_transform(self.nu_dir, self.main_bbox, self.time_info.tspan)

    def get_inverse_transform(self):
        return get_inverse_nu_transform(self.nu_dir, self.main_bbox, self.time_info.tspan)

    def inverse_nu_transform(self, points):
        return inverse_nu_transform(points, self.nu_dir, self.main_bbox, self.time_info.tspan)

    def adapted_nu_grid(self):
        """
        Generate a grid covering the support of nu for this nu_dir.
        Returns a list of ND grid cell centers.
        """
        dim_grids = []
        for gs in self.grid_sizes():
            tmp = np.linspace(0, 1, gs+1)
            dim_grids.append((tmp[1:] + tmp[:-1]) / 2)
        std_grid = np.array(list(itertools.product(*dim_grids)))

        trmat, offset = self.get_transform()
        res = std_grid @ trmat.transpose()
        res += offset

        return res

    def __getitem__(self, idx):
        """Return a BoundingBox representing the grid cell at a given index"""
        return self.nu_bbox.get_grid_cell(self.grid_sizes(), idx)

class MuSpace(RegularProductSpace):
    """Space combining all components of mu, one for each mu_dir"""

    def __init__(self, mu_dirs, grid_size, main_bbox, time_info):
        self.mu_dirs = mu_dirs
        self.num_dirs = len(self.mu_dirs)
        super().__init__(self.num_dirs, (grid_size, grid_size))
        self.grid_size = grid_size
        self.total_points = grid_size ** 2
        self.main_bbox = main_bbox
        self.time_info = time_info

    def grid_sizes(self):
        return self.grid_size, self.grid_size

    def __getitem__(self, mu_idx):
        return MuComponentSpace(self.mu_dirs[mu_idx], self.grid_size, self.main_bbox, self.time_info)

class NuSpace(RegularProductSpace):
    """Space combining all components of nu, one for each nu_dir"""
    def __init__(self, nu_dirs, grid_sizes, main_bbox, time_info):
        self.nu_dirs = nu_dirs
        self.num_dirs = len(self.nu_dirs)
        super().__init__(self.num_dirs, grid_sizes)
        self._grid_sizes = np.asarray(grid_sizes)
        self.total_points = np.prod(grid_sizes)
        self.main_bbox = main_bbox
        self.time_info = time_info

    def __getitem__(self, nu_idx):
        return NuComponentSpace(self.nu_dirs[nu_idx], self._grid_sizes, self.main_bbox, self.time_info)

    @staticmethod
    def dir_from_time(t):
        """
        Get the nu_dir corresponding to a time step t after the
        reparametrization to S^1
        """
        return np.array([1, t]) / math.sqrt(1 + t ** 2)

    @classmethod
    def from_times(cls, times, *args):
        nu_dirs = [cls.dir_from_time(t) for t in times]
        return cls(nu_dirs, *args)


class SnapshotSpace(ShapedSpace, FiniteElementSpace, StrippableSpace):
    """Space representing the snapshot variable u"""

    def __init__(self, grid_sizes, main_bbox, time_info):
        ShapedSpace.__init__(self, grid_sizes)
        self._grid_sizes = np.asarray(grid_sizes)
        self.total_points = np.prod(grid_sizes)
        self.main_bbox = main_bbox
        self.time_info = time_info

    def grid_sizes(self):
        return self._grid_sizes

    def get_transform(self):
        return mpltransforms.BboxTransform(BoundingBox.unit(), self.main_bbox)

    def get_center_grid(self):
        # ordering has to be consistent with nu grids
        # nu grids come from strip projector
        # order of columns in strip projector matrix?
        # nu_component_space.strip_pattern col, cell
        # strip_pattern only 2d
        # matrix col = nu_comp.ravel_index((col, row))
        # np ravel multi (idx, self.grid_sizes())
        dim_grids = []
        for bound, gs in zip(self.main_bbox.bounds, self.grid_sizes()):
            tmp = np.linspace(*bound, gs + 1)
            dim_grids.append((tmp[1:] + tmp[:-1]) / 2)
        return np.array(list(itertools.product(*dim_grids)))

    def __getitem__(self, idx):
        return self.main_bbox.get_grid_cell(self.grid_sizes(), idx)

# TODO: make this useful
class LineProjectionSpace:
    """
    Space for the one-dimensional projections occuring in the
    reduced time consistency constraint
    """

    def __init__(self):
        pass

class PhaseSpace(FiniteElementSpace):
    """Space for the full-dimensional phase space variable lambda"""

    def __init__(self, main_bbox, times, grid_size_per_dim):
        self.main_bbox = main_bbox
        self.time_info = TimeInfo(times)
        self.grid_size_per_dim = np.asarray(grid_size_per_dim)
        self.points_per_dim = np.square(self.grid_size_per_dim)
        self.total_points = np.prod(self.points_per_dim)
        self.dim = len(grid_size_per_dim)

    def get_std_1d_grids(self):
        res = []
        for grid_size in self.grid_size_per_dim:
            grid1d = np.linspace(0, 1, grid_size+1)
            grid1d = (grid1d[1:] + grid1d[:-1]) / 2
            res.append(grid1d)
        return res

    def get_transform(self, dim):
        return get_phase_transform(self.main_bbox[dim], self.time_info.tspan)

    def get_inverse_transform(self, dim):
        return get_inverse_phase_transform(self.main_bbox[dim], self.time_info.tspan)

    def transform(self, points, dim):
        trmat, troffs = self.get_transform(dim)
        return points @ trmat.transpose() + troffs

    def inverse_transform(self, points, dim):
        trmat, troffs = self.get_inverse_transform(dim)
        return (points + troffs) @ trmat.transpose()

    def get_grids(self):
        std_1d_grids = self.get_std_1d_grids()
        res = []
        for dim, std_1d_grid in enumerate(std_1d_grids):
            std_grid = np.array(list(itertools.product(std_1d_grid, std_1d_grid)))
            trmat, offset = self.get_transform(dim)
            phase_grid = offset + std_grid @ trmat.transpose()
            res.append(phase_grid)

        return res

    def mpl_draw(self, axs, *args, **kwargs):
        for dim, (gs, ax) in enumerate(zip(self.grid_size_per_dim, axs)):
            for idx in np.ndindex(gs, gs):
                self.get_dim_cell(idx, dim).mpl_draw(ax, *args, **kwargs)

    def get_dim_cell(self, idx, dim):
        return BoundingBox.unit(2).get_grid_cell(2*[self.grid_size_per_dim[dim]], idx).transformed_affine(*self.get_transform(dim))

    def __getitem__(self, idx):
        idx = np.asarray(idx).reshape(2, -1)
        return [self.get_dim_cell(dim_idx, dim) for dim, dim_idx in enumerate(idx)]

    def grid_sizes(self):
        return  np.repeat(self.grid_size_per_dim, 2)

    def cell_volume(self):
        return np.prod(self.main_bbox.side_lengths() ** 2 / (self.time_info.tspan * self.points_per_dim))

    def ravel_dim_idx(self, idx, dim):
        gs = self.grid_size_per_dim[dim]
        return np.ravel_multi_index(idx, (gs, gs))

    def unravel_dim_idx(self, idx, dim):
        gs = self.grid_size_per_dim[dim]
        return np.unravel_index(idx, (gs, gs))

    def ravel_idx(self, idx):
        idx = np.asarray(idx).reshape(-1, 2)
        # need to return tuple, because ar[[0,1]] is not the same as ar[(0,1)]
        return tuple(self.ravel_dim_idx(dim_idx, dim) for dim, dim_idx in enumerate(idx))

    def grid_coords(self, idx):
        return np.array([grid[i] for grid, i in zip(self.get_grids(), idx)])

    def shape_dims_unravelled(self, array):
        return array.reshape(self.grid_sizes())

    def shape_dims_ravelled(self, array):
        return array.reshape(self.points_per_dim)