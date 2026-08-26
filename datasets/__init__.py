"""
Dataset loading module for UCI datasets.
"""

from .loader import load_dataset, DATASETS
from .credit_card import load_credit_card
from .credit_approval import load_credit_approval
from .german_credit import load_german_credit
from .census_income import load_census_income
from .spambase import load_spambase
from .banknote import load_banknote

__all__ = [
    'load_dataset',
    'DATASETS',
    'load_credit_card',
    'load_credit_approval',
    'load_german_credit',
    'load_census_income',
    'load_spambase',
    'load_banknote'
]