"""
Default of Credit Card Clients Dataset
UCI Machine Learning Repository
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import urllib.request


def load_credit_card():
    """
    Load Default of Credit Card Clients dataset.
    
    Returns
    -------
    tuple : (X, y, n_clusters, dataset_name)
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    
    local_file = "default_credit_card.xls"
    if not os.path.exists(local_file):
        print("  Downloading Credit Card dataset...")
        urllib.request.urlretrieve(url, local_file)
    
    try:
        df = pd.read_excel(local_file, header=1, index_col=0)
    except:
        df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/creditcard.csv')
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X = StandardScaler().fit_transform(X)
    n_clusters = len(np.unique(y))
    
    return X, y, n_clusters, 'Credit Card Default (UCI)'