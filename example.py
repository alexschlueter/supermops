import time

import cvxpy
import matplotlib.pyplot as plt
import numpy as np

from supermops.eval import cluster_weights, match_sources, unbalanced_wstein_sq_cvxpy
from supermops.geometry import BoundingBox
from supermops.models import DynamicSuperresModel, FourierNDModel
from supermops.simulation.generators import uniform_stay_in_square
from supermops.solvers import CVXPySolver
from supermops.spaces import ReducedLinearMotionSpace
from supermops.utils.utils import announce_pickle, min_dyn_sep, move
from supermops.visualization.dimreduc import plot_gt_recons_rows

def main():
    # box domain that particles stay inside of at all measurement times
    main_bbox = BoundingBox([[0, 1], [0, 1]])
    # frequency cutoff
    fc = 2
    # static measurement model takes the truncated Fourier series for cutoff fc
    model_static = FourierNDModel([np.arange(-fc, fc + 1),
                                   np.arange(-fc, fc + 1)])

    # measurement time steps
    K = 1
    times = np.linspace(-1, 1, 2 * K + 1)
    dt = times[1] - times[0]

    # dynamic model concatenates the measurements for all time steps
    model = DynamicSuperresModel(model_static, times=times)

    # Define the ground truth:
    # Support is a list of 4d vectors with one vector per particle.
    # The entries of the vector are [x, y, vx, vy], where (x, y) is the position
    # at time t=0 and vx, vy the components of the velocity for the linear motion
    # true_support = np.array([[0.5, 0.47, 0.2, 0.0],
    #                          [0.5, 0.53, -0.2, 0.0]])
    # true_weights = np.array([1.0, 1.0])

    # or generate randomly:
    true_support, true_weights = uniform_stay_in_square(K, dt, min_num_srcs=6,
                                                        max_num_srcs=6)

    grid_size = 100
    # the space managing the discretization of our dimension-reduced dynamic model
    space = ReducedLinearMotionSpace(
        main_bbox,
        model.times,
        num_rads=grid_size, # number of bins for one-dimensional projections
        num_extra_nu_dirs=3, # extra nu dirs to add to those coming from measurement times
        grid_size_mu=grid_size, # grid size along each dimension for variable mu
        grid_sizes_nu=2*[grid_size], # list of grid sizes for nu, one for each dim
        num_mu_dirs=5 # number of mu_dirs to generate automatically
    )

    print("Building projection matrices for reduced time consistency constraint...")
    redcons_block = space.build_redcons_block(projector="new")
    print("Building measurement matrices...")
    meas_block = space.build_nu_meas_block(model_static)

    target = model.apply(true_support, true_weights)
    noise_lvl = 0.01
    noise_const = 0.2
    # Parameter alpha from the article, inverse used to scale data term
    paper_alpha = noise_const * np.sqrt(noise_lvl)
    rng = np.random.default_rng()
    target += np.sqrt(2 * noise_lvl / target.size) * rng.standard_normal(*target.shape)

    redcons_bound = 0.001 # Bound used in reduced time consistency constraint
    solver = CVXPySolver(space, redcons_block, meas_block,
                        redcons_bound=redcons_bound, paper_alpha=paper_alpha)
    solver.set_target(target)

    print("Start opt...")
    start = time.time()
    mu, nu = solver.solve()
    # mu, nu = solver.solve(use_mosek=False)
    end = time.time()
    print("Finished opt in {:.2f} secs.".format(end - start))
    announce_pickle("sol", (mu, nu), "solution", with_time=True)

    print("Evaluate by clustering neighboring grid weights...")
    detected_per_time = []
    for time_idx in range(len(times)):
        nu_t = space.get_nu_for_time(nu, time_idx)
        detec_support, detec_weights = cluster_weights(nu_t)
        detected_per_time.append(detec_support)

    print("Calculate statistics for t=0:")
    nu_t0 = space.get_nu_for_time(nu, K)
    detec_support, detec_weights = cluster_weights(nu_t0)
    true_support_t0 = move(true_support, 0)
    matched_detec, matched_true = match_sources(detec_support, true_support_t0)
    true_pos = len(matched_detec)
    precision = true_pos / len(detec_support)
    recall = true_pos / len(true_support)
    dyn_sep = min_dyn_sep(true_support, K, dt)
    num_true_sources = len(true_support)
    print(f"Ground truth with {num_true_sources} sources with dynamic separation " +
        f"{dyn_sep:.4f} reconstructed with precision {precision:.3f}, recall " +
        f"{recall:.3f}")

    print("Evaluate by calculating unbalanced Wasserstein distance to ground truth...")
    snapshot_grid = space.snapshot_space.get_center_grid()
    radius = 0.05
    try:
        wstein, _, _ = unbalanced_wstein_sq_cvxpy(true_support_t0, snapshot_grid,
                                                true_weights, nu_t0.flatten(), radius)
        print(f"Unbalanced Wasserstein distance to ground truth at t=0: {wstein:.9f}")
    except cvxpy.error.SolverError:
        print("Solver error. MOSEK not installed?")

    print("Plotting ground truth and reconstruction with marked clusters...")
    plot_gt_recons_rows(true_support, space, nu, detected_per_time)
    plt.show()

if __name__ == "__main__":
    main()