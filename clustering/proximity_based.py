"""
Proximity-Based Divisive Clustering Split.
"""

import numpy as np


def proximity_based_split(dist_matrix):
    """
    Proximity-Based Divisive Clustering Split
    
    Splits a cluster based on direct distance relationships with two maximally
    separated seed points (nearest-seed assignment).
    
    Parameters
    ----------
    dist_matrix : ndarray
        Pairwise distance matrix for the current cluster
        
    Returns
    -------
    ndarray : Binary labels for the split
    """
    n = dist_matrix.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    
    # Select two maximally distant seeds
    ind_0, ind_1 = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    
    child0 = [ind_0]
    child1 = [ind_1]
    remaining = list(range(n))
    remaining.remove(ind_0)
    remaining.remove(ind_1)
    
    if not remaining:
        labels = np.zeros(n, dtype=int)
        labels[child0] = 0
        labels[child1] = 1
        return labels
    
    # Get distances to seeds
    row0 = dist_matrix[ind_0, remaining]
    row1 = dist_matrix[ind_1, remaining]
    
    # Assign to closest seed
    for i, pt in enumerate(remaining):
        if row0[i] < row1[i]:
            child0.append(pt)
        elif row0[i] > row1[i]:
            child1.append(pt)
        else:  # Tie-breaking
            child0.append(pt)
    
    labels = np.zeros(n, dtype=int)
    labels[child0] = 0
    labels[child1] = 1
    return labels