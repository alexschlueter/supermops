import os
from pathlib import Path

from supermops.params.params_v3 import RedlinSettings, param_script_main

gt_root = Path(os.environ["REDLIN_GT_ROOT"])

general = {
    "num_mu_dirs": 0,
    "num_extra_nu_dirs": 0,
    "num_time_steps": 3,
    "grid_size": 100
}

pjobs = []
pjob_id = 1
for gtid in range(1, 2001):
    params = general.copy()
    params.update({
        "tag": "fc2/t3gs100/mu0nu0",
        "pjob_id": pjob_id,
        "gtid": gtid,
        "gt_file": gt_root / f"gt_uniformsep/gt_{gtid}.npy",
        "res_file": f"res/res_{gtid}.npy",
        "fc": 2
    })
    pjobs.append(RedlinSettings(params))
    pjob_id += 1

if __name__ == "__main__":
    param_script_main(pjobs)