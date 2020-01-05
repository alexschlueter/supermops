import time

import numpy as np

from supermops.models import FourierNDModel, DynamicSuperresModel
from supermops.discretization import Discretization
from supermops.eval import cluster_weights, match_sources
from supermops.solvers import SeparableConeSolver
from supermops.utils import announce_save

# define ground truth
support = np.array([[0.5, 0.47, 0.2, 0.0], [0.5, 0.53, -0.2, 0.0]])
weights = np.array([1.0, 1.0])

# static parameters
fc = 1  # frequency cutoff
# static model
model_static = FourierNDModel([np.arange(-fc, fc + 1), np.arange(-fc, fc + 1)])

# dynamic parameters
K = 1  # time steps -K, -K+1, ..., K
dt = 1  # time span between adjacent steps
model = DynamicSuperresModel(model_static, K, dt)

target = model.apply(support, weights)
l1_target = np.linalg.norm(weights, ord=1)

disc = Discretization(
    K=K,
    dt=dt,
    num_mu_dirs=3,  # number of directions for var mu
    num_extra_nu_dirs=3,  # number of directions for var nu (+ time dirs)
    num_rads=100,  # number of grid points for projections
    grid_size_nu=100,  # num of grid points along one axis for nu
    grid_size_mu=100,  # num of grid points along one axis for mu
    adaptive_grids=True,  # whether to generate new grids for each var
)

print("Stage 1: Main objective")
Msys, rhs = disc.build_main_objective(model_static, target)
solver = SeparableConeSolver(disc, Msys, rhs, l1_target, nonneg=False)

print("Start opt...")
start = time.time()
mu, nu = solver.solve()
end = time.time()
print("Finished opt in {:.2f} secs.".format(end - start))

announce_save("sol", (mu, nu), "solution", with_time=True)

los = cluster_weights(nu[-K - 1])
print(los)
tp = len(match_sources(los, support[:, :2])[0])
prec = tp / len(los)
rec = tp / len(support.transpose())
f1 = 2 * prec * rec / (prec + rec)
print(prec, rec, f1)
