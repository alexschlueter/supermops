import numpy as np
import scipy.sparse as sp

from .utils.utils import profile

def siddon(d1, d2, bbox, grid_sizes, eps=1e-5, infinite_line=True, return_intersections=False):
    """Implementation of Siddon's algorithm, DOI: 10.1118/1.595715
    The algorithm calculates the length of intersection of a given line
    (segment) with each grid cell in a regular rectangular grid.

    Args:
        d1: start point of line segment
        d2: end point of line segment
        bbox: BoundingBox of rectangular grid
        grid_sizes: array of grid sizes along each dimension
        eps: Defaults to 1e-5.
        infinite_line: Take an infinite line going through d1, d2. Defaults to True.
        return_intersections: Return the points of intersection of the line with
                              the grid. Defaults to False.

    Returns:
        Array of indices of grid cells with non-empty intersection
        Array of length of the intersection for each index in the first array
    """

    grid_steps = bbox.side_lengths() / grid_sizes
    dims = len(grid_steps)
    line_vec = d2 - d1
    if infinite_line:
        alpha_min = -np.inf
        alpha_max = np.inf
    else:
        alpha_min = 0
        alpha_max = 1
    directions = dims * [1]
    next_alphas = dims * [0.0]
    # consider the line equation l(alpha) = d1 + alpha * (d2 - d1)
    # calculate parameters alpha_min, alpha_max, such that l(alpha_min) is the point where l enters
    # the bounding box and l(alpha_max) is the point where l leaves the bounding box
    for dim in range(dims):
        if abs(line_vec[dim]) < eps:
            # detector positions are approximately equal in this coordinate
            # need to treat this case separately to prevent division by very small number later
            if d1[dim] < bbox.bounds[dim][0] or d1[dim] > bbox.bounds[dim][1]:
                # detector line is (approximately) outside bounding box
                # TODO: symmetry broken between d1, d2
                return [], []
            else:
                # we want to treat this coordinate of the line as constant
                # intersections of the line with grid planes in this coordinate should be ignored later, so
                # we set next_alpha for this coord to the maximum possible value
                next_alphas[dim] = np.inf if infinite_line else 1.0
        else:
            bounds = bbox.bounds[dim]
            # parameter for intersection between l and the plane at the lower bound for this coord
            alpha1 = (bounds[0] - d1[dim]) / line_vec[dim]
            # parameter for intersection between l and the plane at the upper bound for this coord
            alpha2 = (bounds[1] - d1[dim]) / line_vec[dim]
            #print(alpha1, alpha2)

            # swap parameters depending on which is smaller
            if alpha1 < alpha2:
                # take max over params for all coords at beginning of line / min over params for all coords at end of line
                alpha_min = max(alpha_min, alpha1)
                alpha_max = min(alpha_max, alpha2)
            else:
                alpha_min = max(alpha_min, alpha2)
                alpha_max = min(alpha_max, alpha1)
                # remember that we are going in reverse in this dim if we follow the line
                directions[dim] = -1

    idx_per_dim = np.empty(dims, dtype=int)
    # line parameter steps (in each dim) to jump from one grid plane to the next
    alpha_steps = abs(grid_steps / line_vec) #TODO: division by zero?
    for dim in range(dims):
        # calculate grid indices for l(alpha_min), the point where l enters the bounding box
        idx = int((d1[dim] + alpha_min * line_vec[dim] - bbox.bounds[dim][0]) / grid_steps[dim])
        idx = max(0, min(grid_sizes[dim] - 1, idx))
        idx_per_dim[dim] = idx

        # set next_alphas[dim] to the line parameter for the first intersection of l with a grid plane in this dim
        # after alpha_min, i.e. after the line has entered the bounding box
        if next_alphas[dim] < 1.0:
            next_alphas[dim] = (bbox.bounds[dim][0] + idx * grid_steps[dim] - d1[dim]) / line_vec[dim]
            # if we are going in negative direction, the above is already correct due to idx beeing rounded down
            if directions[dim] > 0:
                # otherwise, need to go one step further
                next_alphas[dim] += alpha_steps[dim]

    #print(alpha_min, alpha_max, alpha_steps, idx_per_dim, next_alphas)
    detec_dist = np.linalg.norm(line_vec)
    alpha = alpha_min
    result_idxs = []
    result_lengths = []
    if return_intersections:
        result_inters = []
    # main loop: travel along the line through the bounding box
    while alpha < alpha_max and np.all((0 <= idx_per_dim) & (idx_per_dim < grid_sizes)):
        # get parameter for next intersection with a grid plane in any dimension
        next_alpha = min(next_alphas)
        #print(alpha, next_alpha)
        # add new path element to result
        result_idxs.append(idx_per_dim.copy())
        result_lengths.append((next_alpha - alpha) * detec_dist)
        if return_intersections:
            result_inters.append((d1 + alpha * line_vec, d1 + next_alpha * line_vec))

        for dim in range(dims):
            # check if we (approximately) crossed the next grid plane in this coord
            # could also check for exact equality here, but could then get very small line sections
            # in next step
            if (abs(next_alphas[dim] - next_alpha) < eps):
                next_alphas[dim] += alpha_steps[dim]
                idx_per_dim[dim] += directions[dim]

        # start next iteration from new line parameter
        alpha = next_alpha

    if return_intersections:
        return result_idxs, result_lengths, result_inters
    else:
        return result_idxs, result_lengths

@profile
def cell_strip_intersection(cell, proj_dir, left_border, right_border):
    """Calculate the area of intersection of a grid cell with a given strip.

    Args:
        cell: grid cell
        proj_dir: direction perpendicular to strip
        left_border: offset along proj_dir of left border of strip
        right_border: offset along proj_dir of right border of strip

    Returns:
        intersection area
    """
    inters_area = cell.intersect_halfspace(proj_dir, left_border)
    inters_area -= cell.intersect_halfspace(proj_dir, right_border)
    return inters_area

@profile
def strip_projector_2d(space, proj_dir, rads):
    """Assemble a sparse matrix representing the Radon transform discretized
    using the strip projection method:
    Define bins on the projection direction, then for each bin, cover the 2D
    grid by the strip perpendicular to the proj. dir with boundaries defined by
    the bin boundaries. The area of intersection of each grid cell with this
    strip determines the contribution of the value at this grid cell to the
    value in the 1D bin.

    Args:
        space: StrippableSpace of the discretized variable to be projected
        proj_dir: direction to project onto, perpendicular to the strip
        rads: List of offsets along proj_dir to be used as boundaries of the
              bins and strips

    Returns:
        Sparse scipy matrix
    """
    assert len(proj_dir) == 2
    data, rows, cols = [], [], []
    for row, (left_border, right_border) in enumerate(zip(rads[:-1], rads[1:])):
        for col, cell in space.strip_pattern(proj_dir, left_border, right_border):
            # print(f"row {row}/{len(rads)-1}, col {col}/?")
            inters_area = cell_strip_intersection(cell, proj_dir, left_border, right_border)
            if inters_area > 0:
                data.append(inters_area / cell.area())
                # data.append(inters_area / (right_border - left_border))
                rows.append(row)
                cols.append(col)

    return sp.coo_matrix((data, (rows, cols)))

def center_point_projector(grid, dirs, rads):
    """For each projection direction in dirs, build the Radon matrix which
    performs the projection of the grid points in grid onto the line defined the
    direction. The projection results are discretized onto the 1D grid in rads
    and a linear interpolation is used when the projection of a grid point lies
    between two points in the rad grid.

    Arguments:
        grid {ndarray} -- grid points to project
        dirs {ndarray} -- array of 2d directions to project onto
        rads {ndarray} -- Either an array of 1d grids (one for each dir) or just
            a single 1d grid (use same for all dir)

    Returns:
        list -- Radon matrices for each direction
    """
    dots = dirs @ grid.transpose()
    # proj_inds = np.searchsorted(rads, dots)
    # weights_left = (rads[proj_inds]-dots) / rad_bin_len
    # dd, dg = np.ogrid[0:len(dirs),0:len(grid)]
    # res[dd, proj_inds-1, dg] = weights_left
    # res[dd, proj_inds, dg] = 1-weights_left
    res = []
    for di, d in enumerate(dirs):
        if rads.ndim > 1:
            # got adaptive rad grids, use grid corresponding to proj dir
            rads_for_dir = rads[di]
        else:
            # got oversized rad grid
            rads_for_dir = rads
        proj_inds = np.searchsorted(rads_for_dir, dots[di])
        num_rads = len(rads_for_dir)
        right = proj_inds == num_rads
        left = proj_inds == 0
        mid = ~(right | left)
        left_grid_idc = np.flatnonzero(left)
        right_grid_idc = np.flatnonzero(right)
        mid_grid_idc = np.flatnonzero(mid)
        num_left = len(left_grid_idc)
        num_right = len(right_grid_idc)
        if num_left > 0 and np.max(rads_for_dir[0] - dots[di][left]) > 1e-7:
            print(
                "Warning(build_radon_matrix): Overshooting left rad boundary",
                num_left, np.max(rads_for_dir[0] - dots[di][left]))
        if num_right > 0 and np.max(dots[di][right] - rads_for_dir[-1]) > 1e-7:
            print(
                "Warning(build_radon_matrix): Overshooting right rad boundary",
                num_right, np.max(dots[di][right] - rads_for_dir[-1]))
        proj_inds_mid = proj_inds[mid]
        dots_mid = dots[di][mid]
        rad_bin_len = rads_for_dir[1] - rads_for_dir[0]
        weights_left = (rads_for_dir[proj_inds_mid] - dots_mid) / rad_bin_len
        dir_mat = sp.coo_matrix(
            (weights_left, (proj_inds_mid - 1, mid_grid_idc)),
            (num_rads, len(grid)))
        dir_mat += sp.coo_matrix(
            (1 - weights_left, (proj_inds_mid, mid_grid_idc)),
            (num_rads, len(grid)))
        dir_mat += sp.coo_matrix(
            (
                np.ones_like(left_grid_idc),
                (np.zeros_like(left_grid_idc), left_grid_idc),
            ),
            (num_rads, len(grid)),
        )
        dir_mat += sp.coo_matrix(
            (
                np.ones_like(right_grid_idc),
                (np.full_like(right_grid_idc, num_rads - 1), right_grid_idc),
            ),
            (num_rads, len(grid)),
        )
        # dir_mat = sp.coo_matrix((weights_left[di], (proj_inds[di]-1, np.arange(len(grid)))), (len(rads), len(grid)))
        # dir_mat += sp.coo_matrix((1-weights_left[di], (proj_inds[di], np.arange(len(grid)))), (len(rads), len(grid)))
        res.append(dir_mat)
    return res
