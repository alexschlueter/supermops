import json

import numpy as np

from supermops.geometry import BoundingBox
from supermops.projectors import cell_strip_intersection
from supermops.spaces.variables import MuComponentSpace, NuComponentSpace
from supermops.transforms import adapted_rad_range
from supermops.utils.utils import TimeInfo


main_bbox = BoundingBox([[0, 1], [0, 1]])
times = [-1, 1]
grid_size = 10
grid_size_mu = grid_size
grid_sizes_nu = [grid_size] * 2
mu_dir = (1, 2)
# nu_dir = (1, 2)
nu_dir = (1, 0)
mu_dir /= np.linalg.norm(mu_dir)
nu_dir /= np.linalg.norm(nu_dir)
mu_perp = np.array([mu_dir[1], -mu_dir[0]])
nu_perp = np.array([nu_dir[1], -nu_dir[0]])
#rads = [0.4, 0.6]
#allrads = np.linspace(0, 1, 5)
#rads = allrads[2:4]
num_rads = 5

time_info = TimeInfo(times)
mu_space = MuComponentSpace(mu_dir, grid_size_mu, main_bbox, time_info)
nu_space = NuComponentSpace(nu_dir, grid_sizes_nu, main_bbox, time_info)

rad_range = adapted_rad_range(mu_dir, nu_dir, main_bbox, time_info.tspan)

def write_one_var(dir, proj_dir, grid_size, space):
    d = dict()
    d["dir"] = dir.tolist()
    d["grid_size"] = grid_size
    d["proj_dir"] = proj_dir.tolist()
    d["trmat"], d["troffs"] = map(lambda a: a.tolist(), space.get_transform())
    d["rad_range"] = rad_range
    allrads = np.linspace(*rad_range, num_rads)
    rads = allrads[2:4]
    d["allrads"] = allrads.tolist()
    d["rads"] = rads.tolist()

    d["cells"] = []
    for idx, cell in space.strip_pattern(proj_dir, *rads):
        inters = cell_strip_intersection(cell, proj_dir, *rads) / cell.area()
        fcell = space[space.unravel_index(idx)]
        d["cells"].append({
            "corners": fcell.corners_ccw().tolist(),
            "inters": inters
        })

    return d

res = {
    "mu": write_one_var(mu_dir, nu_dir, grid_size_mu, mu_space),
    "u": write_one_var(nu_dir, mu_dir, grid_size, nu_space)
}

with open("plot_data.json", "w") as file:
    json.dump(res, file, indent=4)
