import time

import numpy as np
import scipy.sparse as sp
from scipy import spatial


def build_radon_matrix(grid, dirs, rads):
    """For each projection direction in dirs, build the radon matrix which
    performs the line projection of the grid points in grid onto the direction.
    The projection results are discretized onto the grid in rads and a linear
    interpolation is used when the projection of a grid point lies between 
    two points in the rad grid.
    
    Arguments:
        grid {ndarray} -- grid points to project
        dirs {ndarray} -- array of 2d directions to project onto
        rads {ndarray} -- Either an array of 1d grids (one for each dir) or just
            a single 1d grid (use same for all dir)
    
    Returns:
        list -- radon matrices for each direction
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


def announce_save(path, obj, descr=None, with_time=False):
    if with_time:
        path += "_" + time.strftime("%Y%m%d-%H%M%S")
    if descr is not None:
        print("Saving {} to {}.npy".format(descr, path))
    else:
        print("Saving {}.npy".format(path))
    np.save(path, obj)


def torus_dist(v, w):
    return np.max(np.minimum(np.abs(v - w), 1 - np.abs(v - w)))


def min_torus_sep(sources):
    return np.min(spatial.distance.pdist(sources, torus_dist))


def min_dyn_sep(support, K, dt):
    min_seps = []
    for k in range(-K, K + 1):
        min_seps.append(min_torus_sep(support[:, :2] +
                                      k * dt * support[:, 2:]))
    return np.min(min_seps)


def dirichlet(grid, support, weights, fc):
    freqs = np.arange(-fc, fc + 1)
    res = 0
    for s, w in zip(support, weights):
        xx = np.outer(grid - s[0], freqs)
        xy = np.outer(grid - s[1], freqs)
        zx = np.exp(-2j * np.pi * xx).sum(axis=1)
        zy = np.exp(-2j * np.pi * xy).sum(axis=1)
        res += w * np.outer(zx, zy)
    return res


def move(support, time):
    return support[:, :2] + time * support[:, 2:]
