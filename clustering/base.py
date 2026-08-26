"""
Base class for divisive hierarchical clustering.
"""

import numpy as np
import heapq
from sklearn.metrics import pairwise_distances


class DivisiveTree:
    """
    Divisive Hierarchical Clustering Tree.
    
    Recursively splits clusters using a specified splitting strategy until
    the desired number of clusters is reached.
    """
    
    def __init__(self, splitter, min_size=2, metric='euclidean'):
        """
        Parameters
        ----------
        splitter : callable
            Function that takes a distance matrix and returns binary labels
        min_size : int
            Minimum cluster size to consider for splitting
        metric : str
            Distance metric ('euclidean', 'cosine', etc.)
        """
        self.splitter = splitter
        self.min_size = min_size
        self.metric = metric
        self.tree = {}
        self.next_id = 0
        self.D = None
    
    def _compute_distance_matrix(self, X):
        """Compute pairwise distance matrix for the dataset."""
        return pairwise_distances(X, metric=self.metric)
    
    def fit(self, X, desired_clusters=2):
        """
        Fit the divisive clustering tree.
        
        Parameters
        ----------
        X : ndarray
            Input data
        desired_clusters : int
            Target number of clusters
            
        Returns
        -------
        ndarray : Cluster labels
        """
        n = X.shape[0]
        self.D = self._compute_distance_matrix(X)
        
        # Initialize root
        root = self._new_node(list(range(n)))
        heap = [(-len(self.tree[root][0]), root)]
        clusters = 1
        
        while heap and clusters < desired_clusters:
            neg_size, node_id = heapq.heappop(heap)
            size = -neg_size
            indices, _, _ = self.tree[node_id]
            
            if size < self.min_size * 2:
                continue
            
            # Split the cluster
            sub_indices = indices
            subD = self.D[np.ix_(sub_indices, sub_indices)]
            
            split_labels = self.splitter(subD)
            
            if len(np.unique(split_labels)) < 2:
                continue
            
            left_idx = [sub_indices[i] for i, lab in enumerate(split_labels) if lab == 0]
            right_idx = [sub_indices[i] for i, lab in enumerate(split_labels) if lab == 1]
            
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue
            
            # Create child nodes
            left_id = self._new_node(left_idx)
            right_id = self._new_node(right_idx)
            self.tree[node_id] = (indices, left_id, right_id)
            
            heapq.heappush(heap, (-len(left_idx), left_id))
            heapq.heappush(heap, (-len(right_idx), right_id))
            clusters += 1
        
        # Assign labels
        labels = np.zeros(n, dtype=int)
        leaves = self._get_leaves()
        for cid, node_id in enumerate(leaves):
            for idx in self.tree[node_id][0]:
                labels[idx] = cid
        return labels
    
    def _new_node(self, indices):
        nid = self.next_id
        self.tree[nid] = (indices, None, None)
        self.next_id += 1
        return nid
    
    def _get_leaves(self):
        return [nid for nid, (_, left, right) in self.tree.items() 
                if left is None and right is None]