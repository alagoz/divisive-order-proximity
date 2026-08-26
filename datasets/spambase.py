"""
Spambase Dataset
UCI Machine Learning Repository
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_spambase():
    """
    Load Spambase dataset.
    
    Returns
    -------
    tuple : (X, y, n_clusters, dataset_name)
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    names = [f'feature_{i}' for i in range(1, 58)] + ['target']
    
    df = pd.read_csv(url, header=None, names=names)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X = StandardScaler().fit_transform(X)
    n_clusters = len(np.unique(y))
    
    return X, y, n_clusters, 'Spambase (UCI)'