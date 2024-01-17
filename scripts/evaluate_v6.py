import time

total_start = time.time()

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from supermops.eval import *
from supermops.geometry import BoundingBox
from supermops.params.params_v3 import (ADCGSettings, RedlinSettings,
                                      add_pjob_args, load_pjobs)
from supermops.spaces.variables import SnapshotSpace
from supermops.utils.utils import *

logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG,
    stream=sys.stdout
)
log = logging.getLogger()

match = re.fullmatch("evaluate_v(\d+).py", Path(__file__).name)
assert match
my_version = match[1]
print(f"I am eval version {my_version}")


parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

parser.add_argument("--eval-dir", default="eval")
parser.add_argument("--eval-name", default="eval_")
parser.add_argument("--skip-missing", action="store_true")
add_pjob_args(parser)

subparsers = parser.add_subparsers(dest="mode")
redlin_parser = subparsers.add_parser("redlin")
RedlinSettings.add_params_to_parser(redlin_parser)
adcg_parser = subparsers.add_parser("adcg")
ADCGSettings.add_params_to_parser(adcg_parser)
# parser.add_argument("--adcg-mode", action="store_true")

args = parser.parse_args()
logging.info(args)
if args.mode == "adcg":
    settings = ADCGSettings
else:
    settings = RedlinSettings
pjobs = load_pjobs(args, settings)

parmfolder = Path(".")
evalfolder = parmfolder / args.eval_dir
evalfolder.mkdir(exist_ok=True)

def convert_np(array):
    # return array.tolist()
    return array

log_hdl = None
for it, pjob in enumerate(pjobs):
    pjob_id = pjob["pjob_id"]
    adcg_mode = isinstance(pjob, ADCGSettings)
    if log_hdl is not None:
        log.removeHandler(log_hdl)
        log_hdl.close()
    try:
        Path(pjob["eval_log"]).parent.mkdir(parents=True, exist_ok=True)
        log_hdl = logging.FileHandler(pjob["eval_log"])
        log.addHandler(log_hdl)
    except KeyError:
        log.info("No eval_log parameter given, logging only to stdout")

    log.info("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    log.info(f"Job {args.jobid} in it {it+1}/{len(pjobs)} working on pjob {pjob_id}:")
    log.info(pjob)
    start = time.time()

    if adcg_mode:
        log.info("ADCG MODE")
    else:
        log.info("GRID MODE")

    num_time_steps = pjob["num_time_steps"]
    times = np.linspace(-1, 1, num_time_steps)
    log.info(f"Using times = {times}")

    gtdat = np.load(pjob["gt_file"], allow_pickle=True)
    support, weights = gtdat[-2:]
    gtsup_t0 = support[:,:2]

    recons_file = Path(pjob["res_file"])
    recons_dict = None
    try:
        if adcg_mode:
            try:
                _, recons_sup, recons_weights = np.load(recons_file, allow_pickle=True)
            except ValueError:
                recons_dict = np.load(recons_file, allow_pickle=True)
                recons_sup = recons_dict["recons_sup"]
                recons_weights = recons_dict["recons_weights"]
        else:
            recons_dict = np.load(recons_file, allow_pickle=True)
            mu, nu = recons_dict["mu"], recons_dict["nu"]

            num_total_nu_dirs, grid_size, _ = nu.shape

            assert num_time_steps % 2 == 1
            idx_t0 = num_time_steps // 2
            bbox = BoundingBox([[0.0, 1.0], [0.0, 1.0]])
            time_info = TimeInfo(times)
            snap_space = SnapshotSpace(2*[grid_size], bbox, time_info)
            snap_grid = snap_space.get_center_grid()
    except FileNotFoundError as error:
        if args.skip_missing:
            continue
        else:
            sys.exit(error)

    end = time.time()
    init_time = end - start
    log.info(f"Finished init in {init_time:.2f} secs.")
    start = time.time()
    log.info("Cluster...")
    # For the ADCG results, we don't actually do any clustering, just filter
    # detected sources by the condition weight > treshold
    eval_times = []
    eval_cluster = []
    eval_cluster_wstein = []
    for ti, t in enumerate(times):
        sources_for_time = move(support, t)
        sep = min_torus_sep(sources_for_time)
        eval_times.append({
            "pjob_id": pjob_id,
            "time": t,
            "sep": sep
        })
        for thresh in [1e-1, 1e-2]:
            if adcg_mode:
                detec_srcs = move(recons_sup, t)
                filter = recons_weights > thresh 
                detec_srcs = detec_srcs[filter]
                detec_weights = recons_weights[filter]
            else:
                detec_srcs, detec_weights = cluster_weights(nu[ti-len(times)], zero_threshold=thresh)
            matched_detec, matched_true = match_sources(detec_srcs, sources_for_time, radius=0.01)
            tp = len(matched_detec)
            cluster_correct = (tp == len(detec_srcs) == len(support))

            eval_cluster.append({
                "pjob_id": pjob_id,
                "time": t,
                "thresh": thresh,
                "true_pos": tp,
                "detec_srcs": convert_np(detec_srcs),
                "detec_weights": convert_np(detec_weights),
                "matched_detec": convert_np(matched_detec),
                "matched_true": convert_np(matched_true),
                "cluster_correct": cluster_correct
            })

            if t == 0.0 and not adcg_mode:
                # wstein to cluster pos
                log.info("Cluster wstein...")
                los_t0 = detec_srcs
                weights_t0 = detec_weights
                for radius in [0.01, 0.05, 0.1]:
                    wstein, _, _ = unbalanced_wstein_sq_cvxpy(gtsup_t0, los_t0, weights, weights_t0, radius)
                    if wstein is None:
                        log.info(f"FAIL cluster {radius}")
                    eval_cluster_wstein.append({
                        "pjob_id": pjob_id,
                        "time": 0.0,
                        "thresh": thresh,
                        "radius": radius,
                        "cluster_sqwstein": wstein
                    })



    log.info("Wstein...")
    if adcg_mode:
        recons_sup_t0 = recons_sup[:,:2]
        recons_weights_t0 = recons_weights
    else:
        recons_sup_t0 = snap_grid
        recons_weights_t0 = nu[-1 - idx_t0].flatten()
    eval_wstein = []
    for radius in [0.01, 0.05, 0.1]:
        wstein, _, _ = unbalanced_wstein_sq_cvxpy(gtsup_t0, recons_sup_t0, weights, recons_weights_t0, radius)
        if wstein is None:
            log.error(f"FAIL {radius}")
        eval_wstein.append({
            "pjob_id": pjob_id,
            "time": 0.0,
            "radius": radius,
            "sqwstein": wstein
        })

    # need times for this!
    dyn_sep = min_dyn_sep(support, times=times)

    end = time.time()
    eval_time = end - start
    log.info(f"Finished eval in {eval_time:.2f} secs.")

    run_dict = {
        "eval_version": my_version,
        "eval_init_time": init_time,
        "eval_time": eval_time,
        **{k: v for k, v in pjob.settings.items() if k != "pjob_id"},
        "support": convert_np(support),
        "weights": convert_np(weights),
        "dyn_sep": dyn_sep,
    }
    if recons_dict is not None:
        run_dict.update({k: v for k, v in recons_dict.items() if k not in ["pjob_id", "params", "mu", "nu", "recons_sup", "recons_weights"]})
    run_series = pd.Series(run_dict, name=pjob["pjob_id"])

    eval_file = evalfolder / f"{args.eval_name}{pjob_id}.pickle"

    all_frames = {
        "run": run_series,
        "time": pd.DataFrame(eval_times),
        "cluster": pd.DataFrame(eval_cluster),
        "wstein": pd.DataFrame(eval_wstein),
    }
    if not adcg_mode:
        all_frames["cluster_wstein"] = pd.DataFrame(eval_cluster_wstein)
    announce_pickle(eval_file, all_frames)

total_end = time.time()
print(f"Total time = {total_end - total_start:.2f} secs")