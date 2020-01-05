import numpy as np
from scipy import ndimage, optimize
from sklearn.neighbors import NearestNeighbors


def cluster_weights(weights, zero_threshold=1e-1, neighborhood=np.ones(
    (3, 3))):
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
    
    Returns:
        list of tuples -- list of source points
    """
    weights[np.abs(weights) < zero_threshold] = 0
    labelled, num_sources = ndimage.label(weights, structure=neighborhood)
    com = np.array(
        ndimage.center_of_mass(weights, labelled, 1 + np.arange(num_sources)))
    com /= np.array(weights.shape) - 1
    return com


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
        return 0

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