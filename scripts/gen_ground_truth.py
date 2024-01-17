import argparse

import numpy as np

from supermops.simulation.generators import uniform_stay_in_square
from supermops.simulation.sepdistr import SeparationDistribution
from supermops.utils.utils import *

parser = argparse.ArgumentParser(
    description="Generate particle configurations which stay in the square " +
    "[0,1]^2. The distribution of dynamic separation values is approximately" +
     " uniform over all generated configurations."
)
# parser.add_argument("--jobid", type=int, default=1)
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--dt", type=float)
parser.add_argument("--num-draws", type=int, default=2000)
parser.add_argument("--num-sep-bins", type=int, default=500)
args = parser.parse_args()
print(args)

if args.dt is None:
    args.dt = 1 / args.K

min_num_srcs = 4
max_num_srcs = 20
# minsep = 0.03
# maxsep = 0.06
wmin = 0.9
wmax = 1.1
sep_range = (0, 0.1)

seps = np.load("seps/separations.npy")
gen = lambda: uniform_stay_in_square(args.K, args.dt, min_num_srcs,
                                     max_num_srcs, wmin, wmax)
distr = SeparationDistribution(gen, args.K, args.dt, seps, sep_range,
                               num_bins=args.num_sep_bins)
print("Loaded distr of {} separations".format(len(seps)))
for it in range(args.num_draws):
    support, weights = distr.rejection_sampling(verbose=True)
    # announce_save("gt_{}".format(it+1), (None, support, weights), "generated sources")
    announce_save("gt_{}".format(it+1), np.array([None, support, weights], dtype=object), "generated sources")
