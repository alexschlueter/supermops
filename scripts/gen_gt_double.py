from supermops.simulation.generators import uniform_stay_in_square
from supermops.utils.utils import *

for gtid in range(1, 501):
    while True:
        support, weights = uniform_stay_in_square(1, 1, min_num_srcs=2,
            max_num_srcs=2)
        if min_dyn_sep(support, K=2, dt=0.5) > 0.4:
            break
    print(gtid, support, weights)
    announce_pickle(f"./gt_double/gt_double_{gtid}", (support, weights), f"gt {gtid}")