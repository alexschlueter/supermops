from supermops.simulation.generators import uniform_stay_in_square
from supermops.utils.utils import *

for gtid in range(1, 501):
    support, weights = uniform_stay_in_square(1, 1, min_num_srcs=1,
        max_num_srcs=1)
    print(support, weights)
    announce_pickle(f"./gt_single/gt_single_{gtid}", (support, weights), f"gt single {gtid}")