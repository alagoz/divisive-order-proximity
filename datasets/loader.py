"""
Dataset loading utilities.
"""

import numpy as np
from .credit_card import load_credit_card
from .credit_approval import load_credit_approval
from .german_credit import load_german_credit
from .census_income import load_census_income
from .spambase import load_spambase
from .banknote import load_banknote


# Dataset registry
DATASETS = {
    'credit_card': load_credit_card,
    'credit_approval': load_credit_approval,
    'german_credit': load_german_credit,
    'census_income': load_census_income,
    'spambase': load_spambase,
    'banknote': load_banknote
}


def load_dataset(name, max_samples=3000):
    """
    Load a dataset by name.
    
    Parameters
    ----------
    name : str
        Dataset name: 'credit_card', 'credit_approval', 'german_credit',
                     'census_income', 'spambase', 'banknote'
    max_samples : int
        Maximum samples to use (sampled randomly if dataset is larger)
        
    Returns
    -------
    tuple : (X, y, n_clusters, dataset_name)
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    
    X, y, n_clusters, display_name = DATASETS[name]()
    
    # Further subsample if needed
    if X.shape[0] > max_samples:
        idx = np.random.RandomState(42).choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
        y = y[idx]
    
    return X, y, n_clusters, display_name