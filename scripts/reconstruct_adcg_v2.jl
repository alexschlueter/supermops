import Dates
import InteractiveUtils
println("Julia entry ", Dates.now())
println(InteractiveUtils.versioninfo())

using DynSpikeSuperRes.SuperResModels
using SparseInverseProblemsMod
using DynSpikeSuperRes.Utils

using PyCall
println("PyCall pyversion = $(PyCall.pyversion), program = $(PyCall.pyprogramname)")
@pyimport numpy as np
println("After imports ", Dates.now())

py"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import logging
from pathlib import Path
import re
import sys

import pandas as pd

from supermops.params.params_v3 import ADCGSettings, add_pjob_args, load_pjobs

stream_hdl = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG,
    handlers=[
        stream_hdl
    ]
)
log = logging.getLogger()

match = re.fullmatch("reconstruct_adcg_v(\d+).jl", Path($PROGRAM_FILE).name)
assert match
my_version = match[1]
log.info(f"I am recons adcg version {my_version}")

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
add_pjob_args(parser)
ADCGSettings.add_params_to_parser(parser)
args = parser.parse_args(args=$ARGS)
log.info(args)
pjobs = load_pjobs(args, ADCGSettings)

log_hdl = None
"""

for (it, pjob) in enumerate(py"pjobs")
    py"""
    it, pjob = $it - 1, $pjob
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
    log.removeHandler(stream_hdl)
    """

    original_stdout = stdout
    out_rd, out_wr = redirect_stdout()
    out_reader = @async begin
        for line in eachline(out_rd)
            println(original_stdout, line)
            py"log.info($line)"
        end
    end

    pjob = PyDict(pjob)
    pjob_id = pjob["pjob_id"]
    _, actual_sources, weights = np.load(pjob["gt_file"], allow_pickle=true)
    jthetas = copy(transpose(convert(Array{Float64}, actual_sources)))
    jweights = convert(Array{Float64}, weights)

    @assert isodd(pjob["num_time_steps"])
    K = div(pjob["num_time_steps"] - 1, 2)
    tau = 1 / K
    fc = pjob["fc"]
    space_grid_size = pjob["space_grid_size"]
    vel_grid_size = pjob["vel_grid_size"]
    v_max = 1 / sqrt(2)
    filter = ones(2*fc+1, 2*fc+1)
    model_static = SuperResModels.Fourier2d(1.0, 1.0, filter, space_grid_size, space_grid_size)
    model_dynamic = SuperResModels.DynamicFourier2d(model_static, v_max, tau, K, vel_grid_size)


    target = phi(model_dynamic, jthetas, jweights)
    sigma = sqrt(2 * pjob["noise_lvl"] / length(target))
    target = Utils.generate_target(model_dynamic, jthetas, jweights, sigma)
    # mkpath("targets")
    # np.save("targets/target_$jobid.npy", (py"None", target))

    if pjob["save_state"]
        mkpath("state")
    end
    resumedata = nothing
    if pjob["resume"]
        try
            resumedata = PyDict(np.load("state/state_$(pjob_id).npy", allow_pickle=true)[1])
            println(resumedata)
            resume_iter = resumedata["iter"]
            println("Resuming from iter $(resume_iter)")
        catch e
            println("Loading resume file failed")
            resumedata = nothing
        end
    end

    run_start = Dates.now()
    println("Before simul run ", run_start)
    thetas_est, weights_est = Utils.run_simulation_target(pjob_id,
        model_dynamic, jthetas, jweights, target, pjob["max_outer_iter"],
        min_obj_progress=pjob["min_obj_progress"],
        save_state=pjob["save_state"], resume=resumedata)
    opt_time = Dates.value(Dates.now() - run_start) / 1000
    if !isnothing(resumedata)
        opt_time += resumedata["opt_time"]
    end

    thetas_est = transpose(thetas_est)
    println(opt_time, thetas_est, weights_est)

    py"""
    try:
        import resource
        maxrss_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
        maxrss_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1000
        print(f"{maxrss_self:.2f} MB, {maxrss_children:.2f} MB")
        maxrss = maxrss_self + maxrss_children
        print(f"{maxrss:.2f} MB")
    except ModuleNotFoundError:
        maxrss_self, maxrss_children = None, None

    Path(pjob["res_file"]).parent.mkdir(parents=True, exist_ok=True)
    pd.Series({
        "recons_adcg_version": my_version,
        **pjob.settings,
        "recons_sup": $thetas_est,
        "recons_weights": $weights_est,
        "opt_time": $opt_time,
        "maxrss_self_allpjobs": maxrss_self,
        "maxrss_children_allpjobs": maxrss_children,
    }).to_pickle(pjob["res_file"])
    """

    redirect_stdout(original_stdout)
    close(out_wr)
    fetch(out_reader)
    py"log.addHandler(stream_hdl)"
end
