import numpy as np
from matplotlib import transforms
from matplotlib.lines import Line2D


def overlapping_extent(bbox, grid_sizes):
    grid_sizes = np.asarray(grid_sizes)
    overlap = bbox.side_lengths() / (2 * (grid_sizes - 1))
    return bbox.padded(overlap)

def draw_grid_sol(ax, weights, transform=transforms.IdentityTransform(), **kwargs):
    # extent = overlapping_extent(bbox, weights.shape).flatten()
    img = ax.imshow(weights.transpose(),
                    interpolation="none",
                    origin="lower",
                    aspect="auto",
                    extent=[0, 1, 0, 1],
                    transform=transform + ax.transData,
                    **kwargs)
    # img.set_extent([-3,1,0,1])

    bbox = transforms.Bbox.from_extents(0, 0, 1, 1)
    trbox = transforms.Bbox.null()
    trbox.update_from_data_xy(transform.transform(bbox.corners()))
    ax.set_xlim(trbox.x0, trbox.x1)
    ax.set_ylim(trbox.y0, trbox.y1)

    return img

def draw_grid(ax, bbox, grid_sizes):
    for dim, gs in enumerate(grid_sizes):
        for tick in range(gs + 1):
            pos = bbox.bounds[dim][0] + tick * bbox.side_lengths()[dim] / gs
            params = []
            for dim2 in range(len(grid_sizes)):
                if dim == dim2:
                    params.append([pos, pos])
                else:
                    params.append(bbox.bounds[dim2])
            ax.add_line(Line2D(*params))