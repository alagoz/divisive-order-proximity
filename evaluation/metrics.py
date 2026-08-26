"""
Clustering evaluation metrics.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def evaluate_clustering(y_true, y_pred, X):
    """
    Compute multiple clustering evaluation metrics.
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    X : ndarray
        Data matrix
        
    Returns
    -------
    dict : Dictionary of metrics
    """
    metrics = {
        'ARI': adjusted_rand_score(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred),
    }
    
    if len(np.unique(y_pred)) > 1:
        metrics['Silhouette'] = silhouette_score(X, y_pred)
    else:
        metrics['Silhouette'] = np.nan
    
    return metrics