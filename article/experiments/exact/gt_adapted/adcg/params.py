import os
from pathlib import Path

from supermops.params.params_v3 import ADCGSettings, param_script_main

gt_root = Path(os.environ["REDLIN_GT_ROOT"])

general = {
    "fc": 2,
    "save_state": True,
    "resume": True
}

pjobs = []
pjob_id = 1
for t in [5, 7]:
    for gtid in range(1, 2001):
        params = general.copy()
        params.update({
            "tag": "2021-11-01-new-dynsep-adcg",
            "num_time_steps": t,
            "pjob_id": pjob_id,
            "gtid": gtid,
            "gt_file": gt_root / f"gt_uniformsep/other/t{t}/gt_{gtid}.npy",
            "res_file": f"res/t{t}/res_{gtid}.pickle",
            "recons_log": f"log/t{t}/out_{gtid}.log",
            "eval_log": f"log/eval/pjob_{pjob_id}.log"
        })
        pjobs.append(ADCGSettings(params))
        pjob_id += 1

if __name__ == "__main__":
    param_script_main(pjobs)