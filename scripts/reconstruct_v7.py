from datetime import datetime
print(f"Entry: {datetime.now()}")

import time
total_start = time.time()

import argparse
from contextlib import redirect_stdout, redirect_stderr
import logging
from pathlib import Path
import re
import sys

import numpy as np
from numpy.random import default_rng
import pandas as pd
import psutil

from supermops.geometry import BoundingBox
from supermops.models import FourierNDModel, DynamicSuperresModel
from supermops.params.params_v3 import RedlinSettings, add_pjob_args, load_pjobs
from supermops.spaces.reducedlinear import ReducedLinearMotionSpace
from supermops.solvers.cvxpy_solver import CVXPySolver

def main():
    print(f"After imports: {datetime.now()}")
    print(f"Affinity {len(psutil.Process().cpu_affinity())}")
    match = re.fullmatch("reconstruct_v(\d+).py", Path(__file__).name)
    assert match
    my_version = match[1]
    print(f"I am recons version {my_version}")

    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        stream=sys.stdout
        # handlers=[
        #     logging.StreamHandler(sys.stdout)
        # ]
    )
    log = logging.getLogger()
    log.write = lambda msg: log.info(msg) if msg != '\n' else None
    log.flush = lambda: None

    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    add_pjob_args(parser)
    RedlinSettings.add_params_to_parser(parser)
    args = parser.parse_args()
    log.info(args)
    pjobs = load_pjobs(args, RedlinSettings)

    global changed

    def did_change(name):
        return old_params is None or old_params[name] != pjob[name]

    def get_new(name):
        global changed
        if did_change(name):
            changed = True
        return pjob[name]

    rng = default_rng()
    bbox = BoundingBox([[0.0, 1.0], [0.0, 1.0]])

    print(f"Pre loop: {datetime.now()}")
    old_params = None
    log_hdl = None
    for it, pjob in enumerate(pjobs):
        assert isinstance(pjob, RedlinSettings)
        changed = False
        # try:

        if log_hdl is not None:
            log.removeHandler(log_hdl)
            log_hdl.close()
        try:
            Path(pjob["recons_log"]).parent.mkdir(parents=True, exist_ok=True)
            log_hdl = logging.FileHandler(pjob["recons_log"], mode="w")
            log.addHandler(log_hdl)
        except KeyError:
            log.info("No recons_log parameter given, logging only to stdout")

        log.info("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.info(f"Job {args.jobid} in it {it+1}/{len(pjobs)} working on pjob {pjob['pjob_id']}:")
        log.info(pjob)

        start = time.time()

        fc = get_new("fc")
        if changed:
            model_static = FourierNDModel([np.arange(-fc, fc + 1), np.arange(-fc, fc + 1)])

        num_time_steps = get_new("num_time_steps")
        if changed:
            times = np.linspace(-1, 1, num_time_steps)
            model = DynamicSuperresModel(model_static, times=times)

        grid_size = get_new("grid_size")
        num_extra_nu_dirs = get_new("num_extra_nu_dirs")
        num_mu_dirs = get_new("num_mu_dirs")
        if changed:
            space = ReducedLinearMotionSpace(
                bbox,
                model.times,
                num_rads=grid_size,
                num_extra_nu_dirs=num_extra_nu_dirs,
                grid_size_mu=grid_size,
                grid_sizes_nu=2*[grid_size],
                num_mu_dirs=num_mu_dirs
            )

        projector = get_new("projector")
        if changed:
            redcons_block = space.build_redcons_block(projector=projector)
            meas_block = space.build_nu_meas_block(model_static)
        end = time.time()

        noise_lvl = get_new("noise_lvl")
        redcons_bound = get_new("redcons_bound")
        if noise_lvl > 0:
            noise_const = get_new("noise_const")
            if changed:
                paper_alpha = noise_const * np.sqrt(noise_lvl)
                solver = CVXPySolver(space, redcons_block, meas_block, redcons_bound=redcons_bound, paper_alpha=paper_alpha)
        else:
            data_scale = get_new("data_scale")
            if changed:
                solver = CVXPySolver(space, redcons_block, meas_block, redcons_bound=redcons_bound, data_scale=data_scale)

        if did_change("gt_file"):
            gtdat = np.load(pjob["gt_file"], allow_pickle=True)
            support, weights = gtdat[-2:]

        # if changed:
        target = model.apply(support, weights)
        target += np.sqrt(2 * pjob["noise_lvl"] / target.size) * rng.standard_normal(*target.shape)

        solver.set_target(target)
        # else:
        #     raise Exception("Nothing changed!")

        init_time = end - start
        log.info(f"Finished init in {init_time:.2f} secs.")
        log.info("Start opt...")
        start = time.time()
        with redirect_stdout(log), redirect_stderr(log):
            mu, nu = solver.solve()
        end = time.time()
        opt_time = end - start
        log.info(f"Finished opt in {opt_time:.2f} secs.")
        try:
            import resource
            try:
                maxrss = resource.getrusage(resource.RUSAGE_BOTH).ru_maxrss / 1000
            except AttributeError:
                maxrss_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
                maxrss_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1000
                log.info(f"{maxrss_self=:.2f} MB, {maxrss_children=:.2f} MB")
                maxrss = maxrss_self + maxrss_children
            log.info(f"{maxrss=:.2f} MB")
        except ModuleNotFoundError:
            maxrss = None

        Path(pjob["res_file"]).parent.mkdir(parents=True, exist_ok=True)
        pd.Series({
            "recons_version": my_version,
            **pjob.settings,
            "mu": mu,
            "nu": nu,
            "init_time_inac": init_time,
            "opt_time": opt_time,
            "maxrss_allpjobs": maxrss
        }).to_pickle(pjob["res_file"])
        # }).to_hdf(pjob["res_file"], key="recons")

        old_params = pjob
        # except Exception as e:
        #     print(e)

    total_end = time.time()
    print(f"Total time = {total_end - total_start}")
    print(f"Exit: {datetime.now()}")

if __name__ == "__main__":
    main()
