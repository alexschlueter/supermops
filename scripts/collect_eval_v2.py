import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

from supermops.params.params_v3 import add_pjob_args, load_pjobs

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    fromfile_prefix_chars="@",
    description="Collect evaluation results into single file in HDF format, " +
                "concatenating all DataFrames."
)
add_pjob_args(parser)
parser.add_argument("--eval-dir", default="eval")
parser.add_argument("--eval-name", default="eval_")
parser.add_argument("--skip-missing", action="store_true")

args = parser.parse_args()
pjobs = load_pjobs(args)

eval_folder = Path(args.eval_dir)
collect_file = eval_folder / "eval_all.h5"
if collect_file.exists():
    sys.exit("Collect file already exists! Unsure if h5 handles this well. Exiting")

collect = {k: [] for k in [
    # EMrecon
    "params", "stats_per_iter",
    # Redlin / ADCG
    "run", "time", "cluster", "cluster_wstein", "wstein"
]}

for pjob in pjobs:
    pjob_id = pjob["pjob_id"]

    print(f"{pjob_id} / {len(pjobs)}")
    try:
        with open(eval_folder / f"{args.eval_name}{pjob_id}.pickle", "rb") as file:
            eval_dict = pickle.load(file)

        for k in eval_dict.keys():
            collect[k].append(eval_dict[k])
    except FileNotFoundError as error:
        print(error)
        if not args.skip_missing:
            sys.exit(-1)

def store(dat, **kwargs):
    dat.to_hdf(collect_file, complevel=9, **kwargs)

for k, v in collect.items():
    if v:
        if k in ["params", "run"]:
            df = pd.DataFrame(v)
            df.index.name = "pjob_id"
        else:
            df = pd.concat(v, ignore_index=True)
        store(df, key=k)