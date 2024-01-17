import argparse

from supermops.simulation.generators import *
from supermops.simulation.sepdistr import *
from supermops.utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--jobid", type=int, default=1)
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--dt", type=float)
parser.add_argument("--num-sep-draws", type=int, default=100_000)
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

assert args.num_sep_draws % 10 == 0

gen = lambda: uniform_stay_in_square(args.K, args.dt, min_num_srcs, max_num_srcs, wmin, wmax)
for i in range(10):
    seps = gen_separation_distr(gen, args.K, args.dt, num=args.num_sep_draws // 10, verbose=True)
    announce_save(f"sep_{args.jobid}_{i+1}", seps, "seps")
