import numpy as np
from matplotlib import transforms, pyplot as plt

from supermops.geometry import BoundingBox
from supermops.utils.utils import move

from .plot import draw_grid_sol


def matrix_offset_to_mpl_affine(matrix, offset):
    assert matrix.shape == (2, 2)
    assert offset.shape == (2,)
    affine_mat = np.block([[matrix, offset[:,None]],
                          [0, 0, 1]])
    return transforms.Affine2D(affine_mat)

def plot_phase_dim(ax, space, weights, dim, **kwargs):
    tr = matrix_offset_to_mpl_affine(*space.get_transform(dim))
    return draw_grid_sol(ax, weights, tr, **kwargs)

def plot_mu(ax, space, weights, **kwargs):
    weights = space.reshape(weights)
    tr = matrix_offset_to_mpl_affine(*space.get_transform())
    return draw_grid_sol(ax,  weights, tr, **kwargs)

def plot_nu(ax, space, weights, dims=[0,1], **kwargs):
    weights = space.reshape(weights)
    matrix, offset = space.get_transform()
    tr = matrix_offset_to_mpl_affine(matrix[np.ix_(dims, dims)], offset[dims])
    return draw_grid_sol(ax, weights, tr, **kwargs)

def plot_snapshot(ax, weights, bbox=BoundingBox.unit(2), dims=[0,1], **kwargs):
    tr = transforms.BboxTransformTo(bbox.to_mpl_bbox())
    return draw_grid_sol(ax, weights, tr, **kwargs)

def plot_joint_radon(ax, true_sources, mu_dir, **kwargs):
    num_sources = len(true_sources)
    true_sources = np.asarray(true_sources).reshape(num_sources, 2, -1)
    for s in true_sources:
        joint_radon = [mu_dir @ s[0], mu_dir @ s[1]]
        ax.plot([joint_radon[0]], [joint_radon[1]], **kwargs)

def plot_mu_gt(*args, **kwargs):
    plot_joint_radon(*args, **kwargs, marker="o", fillstyle="none")

def plot_mu_los(*args, **kwargs):
    plot_joint_radon(*args, **kwargs, marker="x")

def plot_all_mu_sols(axs, space, true_sources, mu, **kwargs):
    mu = space.reshape(mu)
    for di in range(space.num_dirs):
        plot_mu(axs[di], space[di], mu[di], **kwargs)
        plot_mu_gt(axs[di], true_sources, space.mu_dirs[di], **kwargs)

def plot_nu_gt(ax, true_sources, nu_dir, dims=[0,1], **kwargs):
    num_sources = len(true_sources)
    true_sources = np.asarray(true_sources).reshape(num_sources, 2, -1)
    true_sources = true_sources[:,:,dims]
    for s in true_sources:
        point = nu_dir[0] * s[0] + nu_dir[1] * s[1]
        ax.plot(*point, marker="o", fillstyle="none", **kwargs)

def plot_all_nu_sols(axs, space, true_sources, nu, **kwargs):
    nu = space.reshape(nu)
    for di in range(space.num_dirs):
        plot_nu(axs[di], space[di], nu[di], **kwargs)
        plot_nu_gt(axs[di], true_sources, space.nu_dirs[di], **kwargs)

def plot_snapshot_gt(ax, true_sources, time, dims=[0,1], arrows=True, arscale=0.5, **kwargs):
    num_sources = len(true_sources)
    true_sources = np.asarray(true_sources).reshape(num_sources, 2, -1)
    true_sources = true_sources[:,:,dims]
    pos = move(true_sources.reshape(num_sources, -1), time)
    # pos = move(true_sources, time - space.time_info.tmid)
    # print(pos)
    for p in pos:
        ax.plot(*p, **kwargs)
    # ax.scatter(pos[:, 0], pos[:, 1], **kwargs)
    if arrows:
        for i in range(len(true_sources)):
            ax.annotate("",
                        xy=pos[i] + true_sources[i, 1] * arscale,
                        xytext=pos[i],
                        arrowprops=dict(arrowstyle="->", color='m'),
                        annotation_clip=False)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

def plot_gt_recons_rows(true_support, space, nu, detected):
    fig, (axs_gt, axs_recon) = plt.subplots(2,
                                                      space.time_info.num_times,
                                                      figsize=(10, 10))
    #, sharey="row")#, sharex="col")
    axs_gt[0].set_ylabel("Ground Truth")
    axs_recon[0].set_ylabel("Reconstruction")

    for time_idx, time in enumerate(space.time_info.times):
        ax_gt = axs_gt[time_idx]
        plot_snapshot_gt(ax_gt, true_support, time, marker="x")
        ax_gt.set_title(f"t = {time}")
        ax_gt.set_xlim(*space.main_bbox.bounds[0])
        ax_gt.set_ylim(*space.main_bbox.bounds[1])
        plot_snapshot(axs_recon[time_idx], space.get_nu_for_time(nu, time_idx), space.main_bbox)
        det_at_time = detected[time_idx]
        axs_recon[time_idx].scatter(det_at_time[:, 0],
                             det_at_time[:, 1],
                             facecolors="none",
                             edgecolors="red")


def interactive_phase_var_plot(fig, axs, phase_space, phase_var, initial_idx=None, con_event="button_press_event", **kwargs):
    # "button_press_event", "motion_notify_event"
    # proj = [0, 1]
    if initial_idx is None:
        cur_idx = phase_space.dim * [0]
    else:
        cur_idx = initial_idx
    phase_var = phase_space.shape_dims_ravelled(phase_var)
    imgs = []
    for dim, (ax, dim_idx, gs) in enumerate(zip(axs, cur_idx, phase_space.grid_size_per_dim)):
        #imgs.append(plot_phase_dim(ax, ss.phase_space, np.sum(phase_var, axis=1-dim).reshape(gs, -1), dim))
        idx = cur_idx.copy()
        idx[dim] = slice(None)
        imgs.append(plot_phase_dim(ax, phase_space, phase_var[tuple(idx)].reshape(gs, gs), dim, **kwargs))
        phase_space.get_dim_cell(phase_space.unravel_dim_idx(dim_idx, dim), dim).mpl_draw(ax)
    print(cur_idx)
    # fig.colorbar(imgs[0])

    text=axs[0].text(0,0, "", va="bottom", ha="left", c="r")
    def redraw(event):
        try:
            if event.inaxes is not None:
                dim = list(axs).index(event.inaxes)
                invpt = phase_space.inverse_transform([event.xdata, event.ydata], dim)
                if np.all(0 <= invpt) and np.all(invpt < 1):
                    grid_size = phase_space.grid_size_per_dim[dim]
                    dim_idx = np.floor(grid_size * invpt).astype(int)
                    cur_idx[dim] = phase_space.ravel_dim_idx(dim_idx, dim)

                    event.inaxes.patches = []
                    phase_space.get_dim_cell(dim_idx, dim).mpl_draw(event.inaxes)

                    for other_dim, (other_img, gs) in enumerate(zip(imgs, phase_space.grid_size_per_dim)):
                        if other_dim != dim:
                            idx = cur_idx.copy()
                            idx[other_dim] = slice(None)
                            other_img.set_data(phase_var[tuple(idx)].reshape(gs, gs).transpose())
        except Exception as e:
            text.set_text(str(e))

    conid = fig.canvas.mpl_connect(con_event, redraw)

    return imgs, conid
