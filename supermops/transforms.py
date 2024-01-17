import itertools
import numpy as np

from supermops.geometry import BoundingBox


def calc_mu_params(mu_dir, main_bbox):
    """
    Calculate s+ and s- for expression of parallelogram support for mu grid
    """
    tr_mu_dir = main_bbox.side_lengths() * mu_dir
    spos = np.sum(tr_mu_dir[tr_mu_dir > 0])
    sneg = np.sum(tr_mu_dir[tr_mu_dir < 0])

    return tr_mu_dir, spos, sneg

def adapted_rad_range(mu_dir, nu_dir, main_bbox, tspan):
    """
    Generate bounds for the one-dimensional projection of the mu and nu grids
    for a given mu_dir and nu_dir, i.e. bounds to cover the supports of
    Rd_{nu_dir} mu_{mu_dir} and Rd_{mu_dir} nu_{nu_dir}
    """
    _, spos, sneg = calc_mu_params(mu_dir, main_bbox)
    # corners of the 2D support of mu_{mu_dir}
    mu_corners = [(sneg, 0), (spos, 0), ((spos+sneg)/2, (spos-sneg)/tspan),
                    ((spos+sneg)/2, (sneg-spos)/tspan)]
    mu_corners = np.array(mu_corners) + (np.dot(mu_dir, main_bbox.origin()), 0)
    proj_corners = mu_corners @ nu_dir
    lo, hi = np.min(proj_corners), np.max(proj_corners)
    return lo, hi

def get_mu_transform(mu_dir, main_bbox, tspan, vel_offs=None):
    """
    Matrix and offset vector to transform from unit square to support of mu_{mu_dir}
    """
    _, spos, sneg = calc_mu_params(mu_dir, main_bbox)
    s = spos - sneg
    matrix = np.array([[s / 2, s / 2],
                    [-s / tspan, s / tspan]])
    offset = np.array((sneg, 0))
    offset[0] += np.dot(mu_dir, main_bbox.origin())
    if vel_offs is not None:
        offset[1] += np.dot(mu_dir, vel_offs)

    return matrix, offset

def get_inverse_mu_transform(mu_dir, main_bbox, tspan):
    # _, spos, sneg = calc_mu_params(mu_dir, main_bbox)
    # s = spos - sneg
    # matrix = np.array([[1 / s, -tspan / (2 * s)],
    #                    [1 / s, tspan / (2 * s)]])
    # offset = -np.array((sneg, 0)) - (np.dot(mu_dir, main_bbox.origin()), 0)
    matrix, offset = get_mu_transform(mu_dir, main_bbox, tspan)
    return np.linalg.inv(matrix), -offset

def inverse_mu_transform(points, mu_dir, main_bbox, tspan):
    matrix, offset = get_mu_transform(mu_dir, main_bbox, tspan)
    return np.linalg.solve(matrix, (points - offset).transpose()).transpose()

def nu_std_bounds(nu_dir, tspan):
    """
    Bounds of support of nu_{nu_dir} in a single dimension, assuming the
    observation domain main_bbox is the unit cube
    """
    vscal = abs(nu_dir[1]) / tspan
    if nu_dir[0] >= 2 * vscal:
        lo, hi = 0, nu_dir[0]
    elif nu_dir[0] <= -2 * vscal:
        lo, hi = nu_dir[0], 0
    else:
        lo = nu_dir[0] / 2 - vscal
        hi = nu_dir[0] / 2 + vscal

    return lo, hi

def get_nu_bbox(nu_dir, main_bbox, tspan):
    """
    BoundingBox covering the support of nu_{nu_dir}
    """
    lo, hi = nu_std_bounds(nu_dir, tspan)
    lo_bounds = main_bbox.side_lengths() * lo + nu_dir[0] * main_bbox.origin()
    hi_bounds = main_bbox.side_lengths() * hi + nu_dir[0] * main_bbox.origin()
    return BoundingBox(np.vstack((lo_bounds, hi_bounds)).transpose())

def get_nu_transform(nu_dir, main_bbox, tspan):
    """
    Matrix and offset vector to transform from unit cube to support of nu_{nu_dir}
    """
    lo, hi = nu_std_bounds(nu_dir, tspan)
    matrix = np.diag(main_bbox.side_lengths() * (hi - lo))
    offset = main_bbox.side_lengths() * lo
    offset += nu_dir[0] * main_bbox.origin()
    return matrix, offset

def get_inverse_nu_transform(nu_dir, main_bbox, tspan):
    matrix, offset = get_nu_transform(nu_dir, main_bbox, tspan)
    return np.linalg.inv(matrix), -offset

def inverse_nu_transform(points, nu_dir, main_bbox, tspan):
    matrix, offset = get_nu_transform(nu_dir, main_bbox, tspan)
    return np.linalg.solve(matrix, (points - offset).transpose()).transpose()

def get_phase_transform(dim_bound, tspan):
    """
    Matrix and offset vector to transform from unit square to support of phase
    space measure lambda (only d=1?)
    """
    matrix = np.array([[0.5, 0.5],
                        [-1 / tspan, 1 / tspan]])
    matrix *= dim_bound.side_lengths()[0]
    offset = np.array((dim_bound.bounds[0, 0], 0))
    return matrix, offset

def get_inverse_phase_transform(dim_bound, tspan):
    matrix, offset = get_phase_transform(dim_bound, tspan)
    return np.linalg.inv(matrix), -offset

# should be same as proj_x(invmu(jrad(full_phase_grid)))
def normalized_jrad_coords(phase_unit_grids, mu_dir, main_bbox):
    phase_full = np.array(list(itertools.product(*phase_unit_grids)))
    tr_mu_dir, spos, sneg = calc_mu_params(mu_dir, main_bbox)
    s = spos - sneg
    return 1/s * (phase_full @ tr_mu_dir - sneg)

def normalized_move_coords(phase_grids, nu_dir, main_bbox, tspan):
    coords_per_dim = [nu_dir[0] * grid[:,0] + nu_dir[1] * grid[:,1] for grid in phase_grids]
    invmat, invoffs = get_inverse_nu_transform(nu_dir, main_bbox, tspan)
    normalized = [invscal * (coords + invoff) for coords, invscal, invoff in zip(coords_per_dim, np.diag(invmat), invoffs)]
    return normalized
