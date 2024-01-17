# srun /home/a/a_schl57/.pyenv/versions/supermops/bin/python /scratch/tmp/a_schl57/particle_sims/paper/noisy/find-const/fix-scale/reconstruct.py $SLURM_ARRAY_TASK_ID 1 5 3 --num-time-steps 5 --fc 2 --gt-fstring "gt_single/gt_single_{}.pickle" --noise-const 0.2 --noise-lvl 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 8.0 16.0 32.0
from pathlib import Path
import os

import numpy as np

from supermops.params.params_v3 import RedlinSettings, param_script_main

gt_root = Path(os.environ["REDLIN_GT_ROOT"])

general = {
    "fc": 2,
    "num_mu_dirs": 5,
    "num_extra_nu_dirs": 3,
    "num_time_steps": 5,
    "noise_const": 0.2,
    "grid_size": 200
}

noise_lvls = np.logspace(-10, 10, base=2, num=10)
pjobs = []
pjob_id = 1
for gtid in range(1, 101):
    for noise_lvl in noise_lvls:
        params = general.copy()
        params.update({
            "tag": "2021-09-17-gs200-single",
            "pjob_id": pjob_id,
            "gtid": gtid,
            "gt_file": gt_root / f"single/gt_single/gt_single_{gtid}.pickle",
            "res_file": f"res/single/noise_lvl{noise_lvl}/res_{gtid}.pickle",
            "noise_lvl": noise_lvl
        })
        pjobs.append(RedlinSettings(params))
        pjob_id += 1

for gtid in range(1, 101):
    for noise_lvl in noise_lvls:
        params = general.copy()
        params.update({
            "tag": "2021-09-17-gs200-double",
            "pjob_id": pjob_id,
            "gtid": gtid,
            "gt_file": gt_root / f"double/gt_double/gt_double_{gtid}.pickle",
            "res_file": f"res/double/noise_lvl{noise_lvl}/res_{gtid}.pickle",
            "noise_lvl": noise_lvl
        })
        pjobs.append(RedlinSettings(params))
        pjob_id += 1

if __name__ == "__main__":
    param_script_main(pjobs)
