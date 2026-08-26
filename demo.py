#!/usr/bin/env python
"""
Novel Divisive Clustering Demo
==============================

This script demonstrates the two novel divisive hierarchical clustering algorithms
proposed in the paper:

1. Order-Based Divisive Clustering
2. Proximity-Based Divisive Clustering

The user can select from multiple real-world UCI datasets and evaluate the 
clustering performance using standard metrics.

Usage:
    python demo.py --dataset banknote
    python demo.py --dataset credit_card --n_clusters 2
    python demo.py --dataset spambase --metric cosine
    python demo.py --dataset german_credit --max_samples 1000
"""

import time
import argparse
import sys
import os

# Add parent directory to path if running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.cluster import KMeans

# Import from modules
from clustering import DivisiveTree, order_based_split, proximity_based_split
from datasets import load_dataset, DATASETS
from evaluation import plot_results, plot_cluster_metrics
from utils.helpers import print_results, get_summary_table


def run_demo(dataset_name='banknote', n_clusters=None, metric='euclidean', max_samples=3000):
    """
    Run the demo for a specific dataset.
    """
    
    print("\n" + "=" * 70)
    print("NOVEL DIVISIVE CLUSTERING DEMO")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_name}...")
    X, y, true_clusters, display_name = load_dataset(dataset_name, max_samples)
    
    if n_clusters is None:
        n_clusters = true_clusters
    
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  True clusters: {true_clusters}, Target clusters: {n_clusters}")
    print(f"  Distance metric: {metric}")
    
    # Run Order-Based Divisive Clustering
    print("\n" + "-" * 50)
    print("Running Order-Based Divisive Clustering...")
    print("-" * 50)
    
    start = time.time()
    tree_order = DivisiveTree(order_based_split, metric=metric)
    pred_order = tree_order.fit(X, desired_clusters=n_clusters)
    time_order = time.time() - start
    
    print_results("Order-Based", y, pred_order, time_order, X)
    
    # Run Proximity-Based Divisive Clustering
    print("\n" + "-" * 50)
    print("Running Proximity-Based Divisive Clustering...")
    print("-" * 50)
    
    start = time.time()
    tree_prox = DivisiveTree(proximity_based_split, metric=metric)
    pred_prox = tree_prox.fit(X, desired_clusters=n_clusters)
    time_prox = time.time() - start
    
    print_results("Proximity-Based", y, pred_prox, time_prox, X)
    
    # Compare with K-Means baseline
    print("\n" + "-" * 50)
    print("Comparing with K-Means baseline...")
    print("-" * 50)
    
    start = time.time()
    pred_kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X)
    time_kmeans = time.time() - start
    
    print_results("K-Means", y, pred_kmeans, time_kmeans, X)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    times = {'order': time_order, 'prox': time_prox, 'kmeans': time_kmeans}
    summary_df = get_summary_table(y, pred_order, pred_prox, pred_kmeans, times, X)
    
    print("\nPerformance Comparison:")
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 70)
    
    # Visualizations
    # 1. Cluster visualization (2x2 grid)
    plot_results(X, y, pred_order, pred_prox, pred_kmeans, display_name)
    
    # 2. Metrics bar chart
    plot_cluster_metrics(summary_df, display_name)
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='Demo for Novel Divisive Clustering Algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py --dataset banknote
    python demo.py --dataset credit_card --n_clusters 2
    python demo.py --dataset spambase --metric cosine
    python demo.py --dataset german_credit --max_samples 1000
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='banknote',
        choices=list(DATASETS.keys()),
        help=f'Dataset to use (default: banknote)'
    )
    
    parser.add_argument(
        '--n_clusters', '-k',
        type=int,
        default=None,
        help='Number of clusters (default: dataset-specific)'
    )
    
    parser.add_argument(
        '--metric', '-m',
        type=str,
        default='euclidean',
        choices=['euclidean', 'cosine'],
        help='Distance metric (default: euclidean)'
    )
    
    parser.add_argument(
        '--max_samples', '-n',
        type=int,
        default=3000,
        help='Maximum samples to use (default: 3000)'
    )
    
    args = parser.parse_args()
    
    run_demo(
        dataset_name=args.dataset,
        n_clusters=args.n_clusters,
        metric=args.metric,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()