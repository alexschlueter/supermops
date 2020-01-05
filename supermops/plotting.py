import numpy as np
from matplotlib import transforms
import matplotlib.pyplot as plt

from .utils import *


def extent_from_gs(gs):
    real_x = real_y = np.linspace(0, 1, gs)
    dx = (real_x[1] - real_x[0]) / 2.0
    dy = (real_y[1] - real_y[0]) / 2.0
    return [real_x[0] - dx, real_x[-1] + dx, real_y[0] - dy, real_y[-1] + dy]


def mu_transform(theta):
    if theta[1] > 0:
        return np.sum(theta) / np.sqrt(2), (0, 0)
    else:
        return (theta[0] - theta[1]) / np.sqrt(2), (theta[1], 0)


def mu_mpl_transform(theta):
    scal, off = mu_transform(theta)
    return transforms.Affine2D().rotate_deg(-45).scale(scal).translate(*off)


def nu_mpl_transform(dir, tmax):
    vscal = abs(dir[1]) / tmax
    if dir[0] - vscal >= 0:
        return transforms.Affine2D().scale(dir[0])
    else:
        off = 0.5 * (dir[0] - vscal)
        return transforms.Affine2D().scale(vscal).translate(off, off)


def plot_grid_sol(ax, grid_size, weights, transform, **kwargs):
    extent = extent_from_gs(grid_size)
    bbox = transforms.Bbox.from_extents(
        np.array(extent).reshape(2, -1).transpose().flatten())
    trbox = transforms.Bbox.null()
    trbox.update_from_data_xy(transform.transform(bbox.corners()))

    img = ax.imshow(weights.transpose(),
                    interpolation="none",
                    origin="lower",
                    extent=extent,
                    transform=transform + ax.transData,
                    **kwargs)

    ax.set_xlim(trbox.x0, trbox.x1)
    ax.set_ylim(trbox.y0, trbox.y1)

    return img


def plot_mu_sol(ax, disc, weights, dir_idx, **kwargs):
    tr = mu_mpl_transform(disc.mu_dirs[dir_idx])
    return plot_grid_sol(ax, disc.grid_size_mu, weights, tr, **kwargs)


def plot_nu_sol(ax, disc, weights, dir_idx, **kwargs):
    tr = nu_mpl_transform(disc.nu_dirs[dir_idx], disc.tmax)
    return plot_grid_sol(ax, disc.grid_size_nu, weights, tr, **kwargs)


def plot_u_sol(ax, disc, weights, time_step, **kwargs):
    tr = transforms.Affine2D()
    return plot_grid_sol(ax, disc.grid_size_nu, weights, tr, **kwargs)


def plot_jrad(ax, disc, actual_sources, theta_idx, **kwargs):
    theta = disc.mu_dirs[theta_idx]
    for s in actual_sources:
        jrad = [theta @ s[:2], theta @ s[2:]]
        ax.plot([jrad[0]], [jrad[1]], **kwargs)


def plot_mu_gt(*args, **kwargs):
    plot_jrad(*args, **kwargs, marker="o", fillstyle="none")


def plot_mu_los(*args, **kwargs):
    plot_jrad(*args, **kwargs, marker="x")


def plot_all_mu_sols(axs, disc, support, mu, **kwargs):
    for di in range(disc.num_mu_dirs):
        plot_mu_sol(axs[di], disc, mu[di], di, **kwargs)
        plot_mu_gt(axs[di], disc, support, di, **kwargs)


def plot_nu_gt(ax, disc, true_sources, dir_idx, **kwargs):
    dir = disc.nu_dirs[dir_idx]
    for s in true_sources:
        point = dir[0] * s[:2] + dir[1] * s[2:]
        ax.plot(*point, marker="o", fillstyle="none", **kwargs)


def plot_all_nu_sols(axs, disc, support, nu, **kwargs):
    for di in range(disc.num_total_nu_dirs):
        plot_nu_sol(axs[di], disc, nu[di], di, **kwargs)
        plot_nu_gt(axs[di], disc, support, di, **kwargs)


def plot_u_gt(ax, support, time, arrows=True, arscale=0.5, **kwargs):
    pos = move(support, time)
    for p in pos:
        ax.plot(*p, **kwargs)
    # ax.scatter(pos[:, 0], pos[:, 1], **kwargs)
    if arrows:
        for i in range(len(support)):
            ax.annotate("",
                        xy=pos[i] + support[i, 2:] * arscale,
                        xytext=pos[i],
                        arrowprops=dict(arrowstyle="->", color='m'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_fourier_vis(support, weights, disc, fc, nu, detected):
    fig, (axs_gt, axs_meas, axs_recon) = plt.subplots(3,
                                                      2 * disc.K + 1,
                                                      figsize=(10, 10))
    #, sharey="row")#, sharex="col")
    diri_grid = np.linspace(0, 1, 1000)
    axs_gt[0].set_ylabel("Ground Truth")
    axs_meas[0].set_ylabel("Measurement")
    axs_recon[0].set_ylabel("Reconstruction")

    for i in range(2 * disc.K + 1):
        k = i - disc.K
        plot_u_gt(axs_gt[i], support, disc.times[i], marker="x")
        axs_gt[i].set_title("k = {}".format(k))
        z = dirichlet(diri_grid, move(support, disc.times[i]), weights, fc)
        axs_meas[i].imshow(np.real(z).transpose(),
                           origin="lower",
                           cmap="gray",
                           extent=extent_from_gs(1000))
        plot_u_sol(axs_recon[i], disc, disc.nu_sol_for_time(nu, k), k)
        det_at_time = detected[i]
        axs_recon[i].scatter(det_at_time[:, 0],
                             det_at_time[:, 1],
                             facecolors='none',
                             edgecolors="green")
