"""
Order-Based Divisive Clustering Split.
"""

import numpy as np


def order_based_split(dist_matrix):
    """
    Order-Based Divisive Clustering Split
    
    Splits a cluster based on relative rank ordering of objects with respect
    to two maximally separated seed points.
    
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
    
    # Initialize child clusters
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
    
    # Rank ordering - position of each point relative to seeds
    order0 = np.argsort(row0)
    order1 = np.argsort(row1)
    sorted_remaining0 = [remaining[i] for i in order0]
    sorted_remaining1 = [remaining[i] for i in order1]
    
    # Assign based on rank comparison
    processed = set()
    for i, pt in enumerate(sorted_remaining0):
        if pt in processed:
            continue
        try:
            pos1 = sorted_remaining1.index(pt)
        except ValueError:
            continue
        
        # Compare ranks: better rank = smaller position
        if pos1 > i:
            child0.append(pt)  # Better rank with seed0
        elif pos1 < i:
            child1.append(pt)  # Better rank with seed1
        else:
            # Tie-breaking by distance
            if row1[i] >= row0[i]:
                child0.append(pt)
            else:
                child1.append(pt)
        processed.add(pt)
    
    # Handle any remaining points
    leftover = [p for p in remaining if p not in processed]
    if leftover:
        split = len(leftover) // 2
        child0.extend(leftover[:split])
        child1.extend(leftover[split:])
    
    labels = np.zeros(n, dtype=int)
    labels[child0] = 0
    labels[child1] = 1
    return labels