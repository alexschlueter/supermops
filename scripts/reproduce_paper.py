import os
from pathlib import Path
import subprocess
import sys

from supermops.params.params_v3 import load_pjobs_from_json

# Adjust this to the actual number of pjobs you want to reconstruct from each
# experiment. To actually reconstruct everything, change to high number
# num_pjobs_per_experiment = 999999
num_pjobs_per_experiment = 1

repo_root = Path(__file__).resolve().parent.parent
ground_truth_root = repo_root / "article/data/ground_truth"
env = {**os.environ, "REDLIN_GT_ROOT": str(ground_truth_root)}
experiments_root = repo_root / "article/experiments"
script_root = repo_root / "scripts"
alberti_root = repo_root / "dynamic_spike_super_resolution"
py_exe = sys.executable
reconstruct_pyredlin = [py_exe, script_root / "reconstruct_v7.py"]
reconstruct_adcg = ["julia", f"--project={alberti_root}",
                    script_root / "reconstruct_adcg_v2.jl"]
evaluate = script_root / "evaluate_v6.py"
collect_eval = script_root / "collect_eval_v2.py"

job_args = ["--param-json", "params.json",
            "--reduction", str(num_pjobs_per_experiment)]

common_args = {
    "env": env,
    "check": True # comment this line to continue on error
}

print("Experiments:")
experiments = list(experiments_root.glob("**/params.py"))
print(list(map(str, experiments)))

for exp_params in experiments:
    exp_root = exp_params.resolve().parent
    print(f"Experiment: {exp_root}")
    rel_params_py = os.path.relpath(exp_params, exp_root)
    print("Generating params.json...")
    subprocess.run([py_exe, rel_params_py, "--write-json"], cwd=exp_root,
                   **common_args)

    print("Running reconstruction...")
    params_json = exp_root / "params.json"
    pjobs = load_pjobs_from_json(params_json)
    method = pjobs[0]["method"]
    if method == "PyRedLin":
        subprocess.run(reconstruct_pyredlin + job_args,
                       cwd=exp_root, **common_args)
    elif method == "ADCG":
        subprocess.run(reconstruct_adcg + job_args,
                       cwd=exp_root, **common_args)
    else:
        raise Exception("unknown method")

    print("Running evaluation...")
    subprocess.run([py_exe, evaluate] + job_args,
                    cwd=exp_root, **common_args)

    print("Collecting evaluations...")
    subprocess.run([py_exe, collect_eval] + job_args,
                    cwd=exp_root, **common_args)

