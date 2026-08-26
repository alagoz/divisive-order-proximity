"""
Credit Approval Dataset
UCI Machine Learning Repository
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_credit_approval():
    """
    Load Credit Approval dataset.
    
    Returns
    -------
    tuple : (X, y, n_clusters, dataset_name)
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    names = ['A' + str(i) for i in range(1, 16)]
    
    df = pd.read_csv(url, header=None, names=names, na_values='?')
    df = df.dropna()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X = StandardScaler().fit_transform(X)
    n_clusters = len(np.unique(y))
    
    return X, y, n_clusters, 'Credit Approval (UCI)'