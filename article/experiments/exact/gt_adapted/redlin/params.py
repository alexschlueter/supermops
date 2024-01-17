import os
from pathlib import Path

from supermops.params.params_v3 import RedlinSettings, param_script_main

gt_root = Path(os.environ["REDLIN_GT_ROOT"])

general = {
    "grid_size": 100,
    "fc": 2
}

munu_dirs = [(0, 0), (3, 0), (5, 3), (10, 7)]

pjobs = []
pjob_id = 1
for t in [5, 7]:
    for mu, nu in munu_dirs:
        for gtid in range(1, 2001):
            params = general.copy()
            params.update({
                "tag": f"2021-10-20-new-dynsep",
                "num_mu_dirs": mu,
                "num_extra_nu_dirs": nu,
                "num_time_steps": t,
                "pjob_id": pjob_id,
                "gtid": gtid,
                "gt_file": gt_root / f"gt_uniformsep/other/t{t}/gt_{gtid}.npy",
                "res_file": f"res/t{t}/mu{mu}nu{nu}/res_{gtid}.pickle",
                "recons_log": f"log/t{t}/mu{mu}nu{nu}/out_{gtid}.log",
                "eval_log": f"log/eval/pjob_{pjob_id}.log"
            })
            pjobs.append(RedlinSettings(params))
            pjob_id += 1

if __name__ == "__main__":
    param_script_main(pjobs)