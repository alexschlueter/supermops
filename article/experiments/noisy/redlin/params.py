import os
from pathlib import Path

import numpy as np

from supermops.params.params_v3 import RedlinSettings, param_script_main
from supermops.utils.utils import parse_int_set

gt_root = Path(os.environ["REDLIN_GT_ROOT"])

gtid_file = Path("./correct-gtids.txt")
with open(gtid_file, "r") as f:
    gtids = sorted(parse_int_set(f.readlines()[1]))

gtids = gtids[:100]
# print(f"gtids = {gtids}")

noise_lvls = np.logspace(-10, 10, base=2, num=10)
# print(noise_lvls)
pjobs = []
pjob_id = 1
for gtid in gtids:
    for noise_lvl in noise_lvls:
        pjobs.append(RedlinSettings({
            "tag": "2021-09-13-wsteinvsdelta-finer",
            "pjob_id": pjob_id,
            "gtid": gtid,
            "gt_file": gt_root / f"gt_uniformsep/gt_{gtid}.npy",
            "res_file": f"res/noise_lvl{noise_lvl}/res_{gtid}.pickle",
            # "grid_size": 250,
            "grid_size": 200,
            "fc": 2,
            "num_mu_dirs": 5,
            "num_extra_nu_dirs": 3,
            "num_time_steps": 5,
            "noise_const": 0.2,
            "noise_lvl": noise_lvl
        }))
        pjob_id += 1

if __name__ == "__main__":
    param_script_main(pjobs)