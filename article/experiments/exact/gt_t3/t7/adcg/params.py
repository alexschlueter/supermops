import os
from pathlib import Path

from supermops.params.params_v3 import ADCGSettings, param_script_main

gt_root = Path(os.environ["REDLIN_GT_ROOT"])

general = {
    "tag": "fc2/t7gs100/adcg",
    "num_time_steps": 7,
    "fc": 2
}

pjobs = []
pjob_id = 1
for gtid in range(1, 2001):
    params = general.copy()
    params.update({
        "pjob_id": pjob_id,
        "gtid": gtid,
        "gt_file": gt_root / f"gt_uniformsep/gt_{gtid}.npy",
        "res_file": f"res/res_{gtid}.npy"
    })
    pjobs.append(ADCGSettings(params))
    pjob_id += 1

if __name__ == "__main__":
    param_script_main(pjobs)