import itertools
import math

import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle as MplRectangle
from matplotlib.transforms import Bbox as MplBbox

from supermops.utils.utils import  profile


class BoundingBox:
    """
    A rectangular, axis-aligned box in d dimensions.
    """

    def __init__(self, bounds):
        """
        Args:
            bounds: Array of size d x 2, where row i contains the lower and
                    upper boundary of the box in dimension (i+1)
        """
        self.bounds = np.atleast_2d(np.asarray(bounds, dtype=float))

    @classmethod
    def unit(cls, dim):
        return cls(dim * [[0, 1]])

    @classmethod
    def from_origin_and_lengths(cls, offsets, lengths):
        offsets = np.asarray(offsets)
        ends = offsets + lengths
        return cls(np.column_stack((offsets, ends)))

    def origin(self):
        return self.bounds[:,0]

    def side_lengths(self):
        return self.bounds[:, 1] - self.bounds[:, 0]

    def __getitem__(self, val):
        return BoundingBox(self.bounds[val])

    def transformed(self, offset, scale):
        offset = np.asarray(offset)
        return BoundingBox(offset[:,np.newaxis] + scale * self.bounds)

    def transformed_affine(self, matrix, offset):
        tr_corners = self.corners() @ matrix.transpose()
        tr_corners += offset
        return Parallelogram(*tr_corners[[1, 0, 2]])

    def envelop(self, other):
        """Make this box larger to contain other BoundingBox"""
        self.bounds[:,0] = np.minimum(self.bounds[:,0], other.bounds[:,0])
        self.bounds[:,1] = np.maximum(self.bounds[:,1], other.bounds[:,1])

    def padded(self, padding):
        bounds = self.bounds.copy()
        bounds[:,0] -= padding
        bounds[:,1] += padding
        return BoundingBox(bounds)

    def corners(self):
        return np.array(list(itertools.product(*self.bounds)))

    def corners_ccw(self):
        assert self.dim() == 2
        return np.array(list(itertools.product(*self.bounds)))[[0,2,3,1]]

    def sides(self):
        return [BoundingBox(np.delete(self.bounds, d, 0)) for d in range(self.dim())]

    def area(self):
        return np.prod(self.side_lengths())

    def midpoint(self):
        return (self.bounds[:,0] + self.bounds[:,1]) / 2

    @profile
    def get_grid_cell(self, grid_sizes, idx):
        """
        Cover this BoundingBox by a regular rectangular grid and return a
        BoundingBox representing the grid cell at index idx
        """
        idx = np.asarray(idx)
        grid_steps = self.side_lengths() / grid_sizes
        return self.__class__((self.origin() + [idx, idx+1] * grid_steps).transpose())

    def intersect_ray(self, point, vec):
        minParam = math.inf
        for (lo, hi), p, v in zip(self.bounds, point, vec):
            if not (lo <= p <= hi):
                return None
            if v > 0:
                minParam = min(minParam, (hi - p) / v)
            elif v < 0:
                minParam = min(minParam, (lo - p) / v)

        return point + minParam * vec

    def intersect_line(self, point, vec):
        zero = (vec == 0)
        boundsz = self.bounds[zero]
        pointz = point[zero]
        if not np.all(boundsz[:,0] <= pointz) and np.all(pointz <= boundsz[:,1]):
            return []
        nonzero = ~zero
        pointnz = point[nonzero]
        vecnz = vec[nonzero]
        boundsnz = self.bounds[nonzero]
        params = (boundsnz.transpose() - pointnz) / vecnz
        # print("params", params)
        sort = np.argsort(params.flatten())
        # print("sort", sort)
        pre_enter = sort[:len(pointnz)].copy()
        # print("pre_enter", pre_enter)
        pre_enter[pre_enter >= len(pointnz)] -= len(pointnz)
        # print("pre_enter", pre_enter)
        if len(np.unique(pre_enter)) != len(pre_enter):
            return []
        # print(params, sort, len(pointnz), sort[[len(pointnz)-1, len(pointnz)]])
        inters_params = params.flatten()[sort[[len(pointnz)-1, len(pointnz)]]]
        # print("inters_params", inters_params)
        unique = list(set([tuple(point + t * vec) for t in inters_params]))
        return np.array(unique)

        # dim = len(point)
        # inters_params = []
        # for (lo, hi), p, v in zip(self.bounds, point, vec):
        #     if v == 0:
        #         if p < lo or hi < p:
        #             return []
        #     else:
        #         t1 = (lo - p) / v
        #         t2 = (hi - p) / v
        #         inters_params += [t1, t2]
        # mid_inters = len(inters_params) // 2
        # inters_params = sorted(inters_params)[mid_inters-1:mid_inters+1]
        # return list(set([tuple(point + t * vec) for t in inters_params]))

    def intersect_halfspace(self, proj_dir, offset, corner_proj=None, eps=1e-6):
        cell_area = self.area()
        if corner_proj is None:
            corner_proj = sorted(proj_dir @ self.corners().transpose())
        proj_prod = np.prod(proj_dir)

        inters_area = 0
        if offset < corner_proj[0]:
            inters_area = cell_area
        elif offset < corner_proj[1]:
            inters_area = cell_area
            if abs(proj_prod) > eps:
                inters_area -= 0.5 * (offset - corner_proj[0]) ** 2 / proj_prod
        elif offset < corner_proj[2]:
            if abs(proj_dir[0]) < eps:
                inters_area = (corner_proj[2] - offset) * self.side_lengths()[0]
            elif abs(proj_dir[1]) < eps:
                inters_area = (corner_proj[2] - offset) * self.side_lengths()[1]
            else:
                inters_area = (0.5 * (corner_proj[1] - corner_proj[0]) + corner_proj[2] - offset) * (corner_proj[1] - corner_proj[0]) / proj_prod
        elif offset < corner_proj[3] and proj_prod > eps:
            inters_area = 0.5 * (corner_proj[3] - offset) ** 2 / proj_prod

        return inters_area

    def dim(self):
        return len(self.bounds)

    def mpl_draw(self, ax, edgecolor='r', facecolor='none', linewidth=1, **kwargs):
        assert self.dim() == 2
        ax.add_patch(MplRectangle(self.origin(), *self.side_lengths(),
            edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, **kwargs))

    def __str__(self):
        return f"BoundingBox({self.bounds})"

    def to_mpl_bbox(self):
        assert self.dim() == 2
        return MplBbox(self.bounds.transpose())

    @staticmethod
    def horizontal_offset(height, point, vec):
        return point[0] + (height - point[1]) * vec[0] / vec[1]

    def box_strip_pattern(self, grid_sizes, proj_dir, left_border, right_border, eps=1e-6):
        assert self.dim() == 2
        perp_dir = perp(proj_dir)
        pnt_left = left_border * proj_dir
        pnt_right = right_border * proj_dir
        if abs(perp_dir[1]) > eps:
            # row_dir = np.array([1, 0])
            for row in range(grid_sizes[1]):
                # row_mid_pnt = np.array([0, (row + 0.5) / self.grid_size])
                # row_mid_inters_a = intersect_lines(row_mid_pnt, row_dir, invpnt_left, perp_dir)
                # row_mid_inters_b = intersect_lines(row_mid_pnt, row_dir, invpnt_right, perp_dir)
                height = self.bounds[1, 0] + (row + 0.5) * (self.bounds[1, 1] - self.bounds[1, 0]) / grid_sizes[1]
                row_mid_inters_a = self.horizontal_offset(height, pnt_left, perp_dir)
                row_mid_inters_b = self.horizontal_offset(height, pnt_right, perp_dir)
                lo, hi = sorted([row_mid_inters_a, row_mid_inters_b])
                col_start = math.floor(self.grid_size * lo - 0.5 * abs(perp_dir[0] / perp_dir[1]))
                col_end = math.ceil(self.grid_size * hi + 0.5 * abs(perp_dir[0] / perp_dir[1]))
                col_start = max(col_start, 0)
                col_end = min(col_end, self.grid_size)
                # print(col_start, col_end)
                for col in range(col_start, col_end):
                    yield row * self.grid_size + col, self[col, row]
        else:
            lo, hi = sorted([invpnt_left[1], invpnt_right[1]])
            row_start = max(math.floor(self.grid_size * lo), 0)
            row_end = min(math.ceil(self.grid_size * hi), self.grid_size)
            for row in range(row_start, row_end):
                for col in range(self.grid_size):
                    yield row * self.grid_size + col, self[col, row]


class Parallelogram:
    """A parallelogram in 2D space"""

    def __init__(self, p1, p2, p3):
        self.p1 = np.asarray(p1)
        self.p2 = np.asarray(p2)
        self.p3 = np.asarray(p3)

    def corners(self):
        return np.array([
            self.p2,
            self.p1,
            self.p3,
            self.p3 + self.p1 - self.p2
        ])

    def corners_ccw(self):
        return np.array([
            self.p1,
            self.p2,
            self.p3,
            self.p3 + self.p1 - self.p2
        ])

    def midpoint(self):
        return (self.p1 + self.p3) / 2

    def sides(self):
        corners = self.corners()
        vecs = np.roll(corners, 1, axis=0) - corners
        return np.hstack((corners[:,None], vecs[:,None]))

    def area(self):
        return np.abs(np.cross(self.p1 - self.p2, self.p3 - self.p2))

    def intersect_line(self, point, vec):
        sides = self.sides()
        inters = []
        for spnt, svec in sides:
            mat = np.vstack((-vec, svec)).transpose()
            try:
                _, inters_param = np.linalg.solve(mat, point - spnt)
                if 0 <= inters_param <= 1:
                    inters.append(spnt + inters_param * svec)
            except np.linalg.LinAlgError:
                pass

        unique = np.array(list(set([tuple(p) for p in inters])))
        return unique

    def mpl_draw(self, ax, edgecolor='r', facecolor='none', **kwargs):
        ax.add_patch(MplPolygon(self.corners()[[1,0,2,3]],
            linewidth=1, edgecolor=edgecolor, facecolor=facecolor, **kwargs))

def perp(vec):
    return np.array([vec[1], -vec[0]])

def intersect_sphere_ray(radius, point, vec):
    pdotp = np.dot(point, point)
    assert pdotp <= radius ** 2
    vdotv = np.dot(vec, vec)
    pdotv = np.dot(point, vec)
    delta = 4 * pdotv ** 2 - 4 * vdotv * (pdotp - radius ** 2)
    # if delta < 0:
    #     return None
    t = (-2 * pdotv + math.sqrt(delta)) / (2 * vdotv)
    return point + t * vec

@profile
def intersect_lines(point1, vec1, point2, vec2):
    mat = np.vstack((-vec1, vec2)).transpose()
    try:
        inters_param, _ = np.linalg.solve(mat, point1 - point2)
        return point1 + inters_param * vec1
    except np.linalg.LinAlgError:
        return None

def intersect_ray_hyperplane(ray_start, ray_dir, plane_normal, plane_offset):
    scal = np.dot(ray_dir, plane_normal)
    if scal == 0:
        return None

    ray_param = (plane_offset - np.dot(ray_start, plane_normal)) / scal
    return ray_start + ray_param * ray_dir