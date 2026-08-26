"""
Helper utility functions.
"""

import numpy as np
import pandas as pd

# Import from evaluation module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import evaluate_clustering


def print_results(name, y_true, y_pred, elapsed_time, X):
    """
    Print formatted results for a single algorithm.
    
    Parameters
    ----------
    name : str
        Algorithm name
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    elapsed_time : float
        Runtime in seconds
    X : ndarray
        Data matrix
    """
    metrics = evaluate_clustering(y_true, y_pred, X)
    
    print(f"\n{name}:")
    print(f"  ARI:        {metrics['ARI']:.4f}")
    print(f"  NMI:        {metrics['NMI']:.4f}")
    print(f"  Silhouette: {metrics['Silhouette']:.4f}")
    print(f"  Time:       {elapsed_time:.4f}s")


def get_summary_table(y_true, y_pred_order, y_pred_prox, y_pred_kmeans, 
                      times, X):
    """
    Create a summary table of results.
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred_order : ndarray
        Order-Based predictions
    y_pred_prox : ndarray
        Proximity-Based predictions
    y_pred_kmeans : ndarray
        K-Means predictions
    times : dict
        Dictionary of runtimes
    X : ndarray
        Data matrix
    
    Returns
    -------
    pd.DataFrame : Summary table
    """
    metrics_order = evaluate_clustering(y_true, y_pred_order, X)
    metrics_prox = evaluate_clustering(y_true, y_pred_prox, X)
    metrics_kmeans = evaluate_clustering(y_true, y_pred_kmeans, X)
    
    summary_df = pd.DataFrame({
        'Algorithm': ['Order-Based', 'Proximity-Based', 'K-Means'],
        'ARI': [metrics_order['ARI'], metrics_prox['ARI'], metrics_kmeans['ARI']],
        'NMI': [metrics_order['NMI'], metrics_prox['NMI'], metrics_kmeans['NMI']],
        'Silhouette': [metrics_order['Silhouette'], metrics_prox['Silhouette'], metrics_kmeans['Silhouette']],
        'Time (s)': [times['order'], times['prox'], times['kmeans']]
    })
    
    return summary_df