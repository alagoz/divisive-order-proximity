"""
Visualization utilities for clustering results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_results(X, y_true, y_pred_order, y_pred_prox, y_pred_kmeans, dataset_name):
    """
    Visualize clustering results side by side.
    
    Parameters
    ----------
    X : ndarray
        Data matrix
    y_true : ndarray
        True labels
    y_pred_order : ndarray
        Order-Based predictions
    y_pred_prox : ndarray
        Proximity-Based predictions
    y_pred_kmeans : ndarray
        K-Means predictions
    dataset_name : str
        Name of the dataset
    """
    # Reduce to 2D for visualization if needed
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_vis = pca.fit_transform(X)
        title_suffix = " (PCA-reduced)"
    else:
        X_vis = X
        title_suffix = ""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Ground Truth
    axes[0, 0].scatter(X_vis[:, 0], X_vis[:, 1], c=y_true, cmap='viridis', s=15, alpha=0.6)
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    # Order-Based
    axes[0, 1].scatter(X_vis[:, 0], X_vis[:, 1], c=y_pred_order, cmap='viridis', s=15, alpha=0.6)
    axes[0, 1].set_title('Order-Based', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    
    # Proximity-Based
    axes[1, 0].scatter(X_vis[:, 0], X_vis[:, 1], c=y_pred_prox, cmap='viridis', s=15, alpha=0.6)
    axes[1, 0].set_title('Proximity-Based', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    # K-Means
    axes[1, 1].scatter(X_vis[:, 0], X_vis[:, 1], c=y_pred_kmeans, cmap='viridis', s=15, alpha=0.6)
    axes[1, 1].set_title('K-Means', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    plt.suptitle(f'Clustering Results Comparison - {dataset_name}{title_suffix}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_results_comparison(X, y_true, predictions_dict, dataset_name):
    """
    Visualize multiple clustering results in a grid.
    
    Parameters
    ----------
    X : ndarray
        Data matrix
    y_true : ndarray
        True labels
    predictions_dict : dict
        Dictionary mapping algorithm names to predictions
    dataset_name : str
        Name of the dataset
    """
    # Reduce to 2D for visualization if needed
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_vis = pca.fit_transform(X)
        title_suffix = " (PCA-reduced)"
    else:
        X_vis = X
        title_suffix = ""
    
    n_algorithms = len(predictions_dict)
    n_cols = min(3, n_algorithms + 1)
    n_rows = (n_algorithms + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Ground Truth
    axes[0].scatter(X_vis[:, 0], X_vis[:, 1], c=y_true, cmap='viridis', s=15, alpha=0.6)
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Algorithm results
    for idx, (name, y_pred) in enumerate(predictions_dict.items(), 1):
        axes[idx].scatter(X_vis[:, 0], X_vis[:, 1], c=y_pred, cmap='viridis', s=15, alpha=0.6)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    
    # Hide empty subplots
    for idx in range(len(predictions_dict) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Clustering Results Comparison - {dataset_name}{title_suffix}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_cluster_metrics(results_df, dataset_name):
    """
    Plot bar chart comparing clustering metrics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing 'Algorithm', 'ARI', 'NMI', 'Silhouette', 'Time (s)'
    dataset_name : str
        Name of the dataset
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metrics comparison
    metrics = ['ARI', 'NMI', 'Silhouette']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    x = np.arange(len(results_df['Algorithm']))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        axes[0].bar(x + i * width, results_df[metric], width, label=metric, color=colors[i])
    
    axes[0].set_xlabel('Algorithm')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Clustering Metrics - {dataset_name}', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(results_df['Algorithm'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, metric in enumerate(metrics):
        for j, value in enumerate(results_df[metric]):
            if not np.isnan(value):
                axes[0].text(j + i * width, value + 0.01, f'{value:.3f}', 
                           ha='center', va='bottom', fontsize=8)
    
    # Runtime comparison
    axes[1].bar(results_df['Algorithm'], results_df['Time (s)'], 
                color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_xlabel('Algorithm')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title(f'Runtime Comparison - {dataset_name}', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, value in enumerate(results_df['Time (s)']):
        axes[1].text(i, value + 0.01, f'{value:.3f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_dimensionality_reduction(X, y, title="Data Visualization"):
    """
    Visualize data using PCA (2D projection).
    
    Parameters
    ----------
    X : ndarray
        Data matrix
    y : ndarray
        Labels
    title : str
        Plot title
    """
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_vis = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        subtitle = f" (explained variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f})"
    else:
        X_vis = X
        subtitle = ""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap='viridis', s=20, alpha=0.7)
    ax.set_title(f'{title}{subtitle}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()