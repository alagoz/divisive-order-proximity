"""
Census Income Dataset (Adult)
UCI Machine Learning Repository
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_census_income():
    """
    Load Census Income (Adult) dataset.
    
    Returns
    -------
    tuple : (X, y, n_clusters, dataset_name)
    """
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    df = pd.read_csv(url_train, header=None, names=names, skipinitialspace=True)
    df = df.replace('?', np.nan).dropna()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X = StandardScaler().fit_transform(X)
    n_clusters = len(np.unique(y))
    
    return X, y, n_clusters, 'Census Income (UCI)'