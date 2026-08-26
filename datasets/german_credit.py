"""
German Credit Dataset
UCI Machine Learning Repository
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_german_credit():
    """
    Load German Credit dataset.
    
    Returns
    -------
    tuple : (X, y, n_clusters, dataset_name)
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    names = [f'feature_{i}' for i in range(1, 21)] + ['target']
    
    df = pd.read_csv(url, header=None, sep=' ', names=names)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X = StandardScaler().fit_transform(X)
    n_clusters = len(np.unique(y))
    
    return X, y, n_clusters, 'German Credit (UCI)'