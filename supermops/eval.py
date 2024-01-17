import dataclasses as dc
from contextlib import nullcontext

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sp
from scipy import ndimage, optimize
from sklearn.neighbors import NearestNeighbors

def cluster_weights(weights, zero_threshold=1e-1, neighborhood=np.ones(
    (3, 3)), centered=True):
    """Create a list of source points from weights on a grid by clustering
    nearby weights which are different enough from zero. The center of mass
    of each cluster (normalized to [0,1]^2) is returned as source point.

    Arguments:
        weights {ndarray} -- 2D array of weights

    Keyword Arguments:
        zero_threshold {scalar} -- treshold under which weights are regarded
            as zero (default: {1e-1})
        neighborhood {ndarray} -- neighborhood pattern for clustering (cf.
            scipy.ndimage.label) (default: {np.ones((3, 3))})
        centered -- Determines what kind of grid is assumed to calculate centers of mass:
            True: Weights at centers of grid cells, e.g. at coords [0.125, 0.375, 0.625, 0.875] for N=4
            False: Weights at corners of grid cells, e.g. at coords [0, 0.333, 0.667, 1] for N=4

    Returns:
        list of centers of mass per cluster
        list of summed weights per cluster
    """
    weights = weights.copy()
    weights[np.abs(weights) < zero_threshold] = 0
    labelled, num_sources = ndimage.label(weights, structure=neighborhood)
    summed_weights = ndimage.sum(weights, labelled, 1 + np.arange(num_sources))
    com = np.array(
        ndimage.center_of_mass(weights, labelled, 1 + np.arange(num_sources)))
    com = com.reshape(-1, 2) # make sure com has two columns even if empty

    if com.size != 0:
        if centered:
            com += 0.5
            com /= np.array(weights.shape)
        else:
            com /= np.array(weights.shape) - 1

    return com, summed_weights


def match_sources(detected_srcs, true_srcs, radius=0.01):
    """Matches detected sources to true sources within a given radius.
    A maximum cardinality matching is found such that each detected source is
    matched to at most one true source, each true source is matched to at most
    one detected source, and all matched pairs are within the given radius of
    each other.

    Arguments:
        detected_srcs {ndarray} -- Ndetec x 2 array of detected src coords
        true_srcs {[type]} -- Ntrue x 2 array of true src coords

    Keyword Arguments:
        radius {float} -- allowed radius (default: {0.01})

    Returns:
        ndarray -- indices of matched detected sources
        ndarray -- indices of matched true sources
    """
    if len(true_srcs) == 0 or len(detected_srcs) == 0:
        return [], []

    # first find all true sources within the tolerance radius around every
    # detection
    nbrs = NearestNeighbors().fit(true_srcs)
    inds = nbrs.radius_neighbors(detected_srcs,
                                 radius=radius,
                                 return_distance=False)

    # then find an optimal matching between detected and true sources
    # which are close enough to each other
    cost = np.ones((len(detected_srcs), len(true_srcs)))
    for i, ind in enumerate(inds):
        cost[i][ind] = 0
    rowids, colids = optimize.linear_sum_assignment(cost)
    allowed_matches = np.argwhere(cost[rowids, colids] == 0)
    detec_ids = rowids[allowed_matches].flatten()
    true_ids = colids[allowed_matches].flatten()
    return detec_ids, true_ids
    # true_positives = len(rowids) - cost[rowids, colids].sum()
    # return true_positives

def unbalanced_wstein_cvxpy(*args, **kwargs):
    obj, tpmat, nu = unbalanced_wstein_sq_cvxpy(*args, **kwargs)
    return np.sqrt(obj), tpmat, nu

def unbalanced_wstein_sq_cvxpy(u, v, u_weights, v_weights, R, verbose=False):
    """Calculate the squared unbalanced Wasserstein-2 divergence between two
    nonnegative mass distributions u and v of possibly unequal total mass:

    min_nu Wasserstein_2(u, nu)^2 + 0.5 * R^2 * ||v - nu||_1
    = min_(nu, tpmat) tpmat .* d + 0.5 * R^2 * ||v - nu||_1
      s.t. (1,...,1) @ tpmat = nu
           tpmat @ (1,...,1) = u_weights
           tpmat >= 0
           nu >= 0

    where
        nu is a vector of weights representing an intermediate mass distribution
            supported in the union of the supports of u and v
        tpmat is the transport plan for the inner Wasserstein-2 distance
        d is a matrix of pairwise squared Euclidean distances between the points
            in the support of u and the points in the support of nu
        .* is element-wise multiplication

    The problem is formulated as a linear optimization problem using CVXPY and
    solved with the MOSEK backend.

    Args:
        u: List of points in support of mass distribution u
        v: List of points in support of mass distribution v
        u_weights: List of weights for u, one for each support point
        v_weights: List of weights for v, one for each support point
        R: Radius balancing mass transport vs. creation / annihilation
        verbose: Defaults to False.

    Returns:
        Optimal value for unbalanced Wasserstein divergence
        Optimal transport plan for inner Wasserstein-2 distance
        Optimal intermediate distribution nu
    """
    d = sp.spatial.distance.cdist(u, np.vstack((u, v)), 'sqeuclidean')
    tpmat = cp.Variable((len(u), len(u) + len(v)))
    nu = cp.Variable(len(u) + len(v))
    ones_u, ones_nu = np.ones(len(u)), np.ones(nu.size)
    constraints = [ones_u @ tpmat == nu, tpmat @ ones_nu == u_weights, tpmat >= 0, nu >= 0] # redundant due to tpmat pushforward constraint: cp.sum(nu) == np.sum(u_weights)
    if v.size == 0:
        obj = cp.Minimize(cp.sum(cp.multiply(tpmat, d)) + 0.5 * R ** 2 * (cp.norm(nu[:len(u)], 1)))
    else:
        obj = cp.Minimize(cp.sum(cp.multiply(tpmat, d)) + 0.5 * R ** 2 * (cp.norm(nu[:len(u)], 1) + cp.norm(nu[len(u):] - v_weights, 1)))
    prob = cp.Problem(obj, constraints)
    # prob.solve(verbose=False, solver=cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 4})
    # some (rare) convergence problems without the option below:
    prob.solve(verbose=verbose, solver=cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 4, "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-6})
    # print("status:", prob.status)
    # print("optimal value", prob.value)
    # print("optimal tpmat", tpmat.value)
    # print("optimal nu", nu.value)
    if prob.status == "optimal":
        return prob.value, tpmat.value, nu.value
    else:
        return None

@dc.dataclass
class PDEval:
    """A class collecting multiple pandas DataFrames from the evaluation."""

    def empty_frame():
        return dc.field(default_factory=pd.DataFrame)

    # EMrecon
    params: pd.DataFrame = empty_frame()
    stats_per_iter: pd.DataFrame = empty_frame()

    # PyRedlin / ADCG

    # Information about the ground truth, pjob settings / parameters,
    # statistics from reconstruction and evaluation phase
    run: pd.DataFrame = empty_frame()
    # Measurement time steps and minimal particle separation
    time: pd.DataFrame = empty_frame()
    # Evaluation result by the weight clustering method
    cluster: pd.DataFrame = empty_frame()
    # Evaluation result: first do weight clustering, then calculate squared
    # unbalanced Wasserstein divergence between cluster centers and ground truth
    cluster_wstein: pd.DataFrame = empty_frame()
    # Evaluation result: calculate squared unbalanced Wasserstein divergence
    # between reconstucted grid weights and ground truth
    wstein: pd.DataFrame = empty_frame()

    @classmethod
    def from_hdf(cls, file_or_store, subkey="", verbose=True):
        expected = [(f.name, subkey + "/" + f.name) for f in dc.fields(cls)]
        if isinstance(file_or_store, pd.HDFStore):
            context = nullcontext(file_or_store)
        else:
            context = pd.HDFStore(file_or_store)
        with context as store:
            # stored = [k for k in expected if k in store]
            # stored = [k[1:] for k in store.keys()]
            # assert all(k in expected for k in stored)
            res = cls(**{k: store[pth] for k, pth in expected if pth in store})
        if verbose:
            print(f"Loaded {file_or_store}")
        return res

    def to_hdf(self, store, subkey=""):
        my_keys = [f.name for f in dc.fields(self)]
        for key in my_keys:
            store[subkey + "/" + key] = getattr(self, key)

    def cluster_correct(self):
        """
        Returns:
            A DataFrame with columns:
            gtid: id of ground truth example
            dyn_sep: minimum distance between two moving ground thruth particles
                over all measurement time steps
            cluster_correct: Boolean indicating whether the configuration at
                time t=0 was correctly reconstructed as evaluated by the
                weight clustering method with weight threshold 0.1
        """
        # res = self.run[["gtid"]].merge(self.cluster.query("time == 0.0 & thresh == 0.1")[["pjob_id", "cluster_correct"]], on="pjob_id")[["gtid", "cluster_correct"]]
        res = self.run[["gtid", "dyn_sep"]].merge(self.cluster.query(
            "time == 0.0 & thresh == 0.1")[["pjob_id", "cluster_correct"]],
            on="pjob_id")[["gtid", "dyn_sep", "cluster_correct"]]
        assert res["gtid"].is_unique
        # return res.sort_values(by="gtid")#["cluster_correct"].tolist()
        return res.sort_values(by="gtid")#.tolist()

    def append(self, other):
        for fname, fval in dc.asdict(self).items():
            setattr(self, fname, fval.append(getattr(other, fname)))

    def filter(self, query):
        """Filter all DataFrames by a query on the runs."""

        res = PDEval()
        res.run = self.run.query(query, engine="python")
        pjobs = res.run.index
        for thefield in dc.fields(self):
            if thefield.name != "run" and not getattr(self, thefield.name).empty:
                setattr(res, thefield.name, getattr(self, thefield.name).query("pjob_id in @pjobs"))
        return res
