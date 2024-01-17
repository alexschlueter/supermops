import math

import numpy as np

from supermops.geometry import perp
from supermops.utils.utils import profile


class StrippableCell:
    """
    A 2D grid cell in the form of a parallelogram, for which the area of
    intersection with a half space can be calculated. This is yielded by the
    StrippableSpace and used in the strip projector.
    """

    def __init__(self, cell_offs, area, outer_proj_offs, inner_proj_offs, factor, parallel_len):
        """
        Save some parameters which are common for each cell in a grid so that
        they don't have to be recalculated for each cell
        """
        self.cell_offs = cell_offs
        self.cached_area = area
        self.outer_proj_offs = outer_proj_offs
        self.inner_proj_offs = inner_proj_offs
        self.factor = factor
        self.parallel_len = parallel_len

    def area(self):
        return self.cached_area

    @profile
    def intersect_halfspace(self, proj_dir, offset, eps=1e-6):
        total_offs = offset - self.cell_offs

        if total_offs < -self.outer_proj_offs:
            return self.cached_area
        elif total_offs < -self.inner_proj_offs:
            return self.cached_area - 0.5 * (total_offs + self.outer_proj_offs) ** 2 * self.factor
        elif total_offs < self.inner_proj_offs:
            if self.factor == 0:
                return (self.inner_proj_offs - total_offs) * self.parallel_len
            else:
                return (0.5 * (self.outer_proj_offs - self.inner_proj_offs) + self.inner_proj_offs - total_offs) * (self.outer_proj_offs - self.inner_proj_offs) * self.factor
        elif total_offs < self.outer_proj_offs:
            return 0.5 * (self.outer_proj_offs - total_offs) ** 2 * self.factor

        return 0

class StrippableSpace:
    """
    A space for a 2D grid variable where the grid cells are parallelograms.
    Provides functionality to apply the strip projector.
    """

    @staticmethod
    def horizontal_offset(height, point, vec):
        return point[0] + (height - point[1]) * vec[0] / vec[1]

    @profile
    def strip_pattern(self, proj_dir, left_border, right_border, eps=1e-6):
        """Yield all grid cells with nonempty intersection with a given strip.

        Args:
            proj_dir: direction perpendicular to strip
            left_border: offset along proj_dir of left strip border
            right_border: offset along proj_dir of right strip border
        """
        grid_sizes = self.grid_sizes()
        # assuming all cells are translated versions of the cell at [0, 0]
        model_cell = self[0, 0]
        area = model_cell.area()
        row_cell_offs = np.dot(proj_dir, model_cell.midpoint())
        trmat, _ = self.get_transform()
        col_delta = np.dot(proj_dir, trmat @ [1 / grid_sizes[0], 0])
        row_delta = np.dot(proj_dir, trmat @ [0, 1 / grid_sizes[1]])

        corners = model_cell.corners()
        side1 = corners[1] - corners[0]
        side2 = corners[2] - corners[0]
        dot12 = np.dot(proj_dir, side1)
        dot32 = np.dot(proj_dir, side2)
        lo_proj, hi_proj = sorted([abs(dot12), abs(dot32)])
        inner_proj_offs = (hi_proj - lo_proj) / 2
        outer_proj_offs = (hi_proj + lo_proj) / 2
        if abs(dot12) < eps:
            parallel_len = np.linalg.norm(side1)
            factor = 0
        elif abs(dot32) < eps:
            parallel_len = np.linalg.norm(side2)
            factor = 0
        else:
            parallel_len = None
            factor = np.linalg.norm(side1 / dot12 - side2 / dot32)
            # important_point = (model_cell.p1 - model_cell.p2) / dot12 - (model_cell.p3 - model_cell.p2) / dot32
            # factor = math.sqrt(important_point[0] ** 2 + important_point[1] ** 2)

        # info used later by cells to calc area of intersection with halfspace
        cell_args = [area, outer_proj_offs, inner_proj_offs, factor, parallel_len]

        invmat, invoffs = self.get_inverse_transform()
        invdir = invmat @ proj_dir
        invperp = invmat @ perp(proj_dir)
        invinvoffs = invmat @ invoffs
        invpnt_left = left_border * invdir + invinvoffs
        invpnt_right = right_border * invdir + invinvoffs

        if abs(invperp[1]) > eps:
            up_down_cor = 0.5 * abs(invperp[0] / (invperp[1] * grid_sizes[1]))
            # row_dir = np.array([1, 0])
            for row in range(grid_sizes[1]):
                # row_mid_pnt = np.array([0, (row + 0.5) / self.grid_size])
                # row_mid_inters_a = intersect_lines(row_mid_pnt, row_dir, invpnt_left, invperp)
                # row_mid_inters_b = intersect_lines(row_mid_pnt, row_dir, invpnt_right, invperp)
                height = (row + 0.5) / grid_sizes[1]
                row_mid_inters_a = self.horizontal_offset(height, invpnt_left, invperp)
                row_mid_inters_b = self.horizontal_offset(height, invpnt_right, invperp)
                lo, hi = sorted([row_mid_inters_a, row_mid_inters_b])
                col_start = math.floor(grid_sizes[0] * (lo - up_down_cor))
                col_end = math.ceil(grid_sizes[0] * (hi + up_down_cor))
                col_start = max(col_start, 0)
                col_end = min(col_end, grid_sizes[0])
                # print(col_start, col_end)
                cell_offs = row_cell_offs + col_start * col_delta
                for col in range(col_start, col_end):
                    # yield row * self.grid_size + col, self[col, row]
                    # print("row", row, "col", col)
                    yield self.ravel_index((col, row)), StrippableCell(cell_offs, *cell_args)
                    cell_offs += col_delta

                row_cell_offs += row_delta
        else:
            lo, hi = sorted([invpnt_left[1], invpnt_right[1]])
            row_start = max(math.floor(grid_sizes[1] * lo), 0)
            row_end = min(math.ceil(grid_sizes[1] * hi), grid_sizes[1])
            row_cell_offs += row_start * row_delta
            for row in range(row_start, row_end):
                cell_offs = row_cell_offs
                for col in range(grid_sizes[0]):
                    # yield row * self.grid_size + col, self[col, row]
                    yield self.ravel_index((col, row)), StrippableCell(cell_offs, *cell_args)
                    cell_offs += col_delta

                row_cell_offs += row_delta