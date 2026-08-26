"""
Clustering module for novel divisive algorithms.
"""

from .base import DivisiveTree
from .order_based import order_based_split
from .proximity_based import proximity_based_split

__all__ = [
    'DivisiveTree',
    'order_based_split',
    'proximity_based_split'
]