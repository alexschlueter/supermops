import os
from pathlib import Path

import numpy as np

from supermops.params.params_v3 import ADCGSettings, param_script_main
from supermops.utils.utils import parse_int_set

gt_root = Path(os.environ["REDLIN_GT_ROOT"])

gtid_file = Path("./correct-adcg-t5.txt")
with open(gtid_file, "r") as f:
    gtids = sorted(parse_int_set(f.readlines()[0]))

gtids = gtids[:50]
# print(f"gtids = {gtids}")

noise_lvls = np.logspace(-10, 10, base=2, num=10)#[1:]
# print(noise_lvls)
pjobs = []
pjob_id = 1
# ORDER of for loops needs to match julia Iterators.product here:
for noise_lvl in noise_lvls:
    for gtid in gtids:
        pjobs.append(ADCGSettings({
            "tag": "2021-09-15-noisy-wsteinvsdelta-adcg",
            "pjob_id": pjob_id,
            "gtid": gtid,
            "gt_file": gt_root / f"gt_uniformsep/gt_{gtid}.npy",
            "res_file": f"res/noise_lvl{noise_lvl}/res_{gtid}.npy",
            "fc": 2,
            "num_time_steps": 5,
            "noise_lvl": noise_lvl
        }))
        pjob_id += 1

if __name__ == "__main__":
    param_script_main(pjobs)