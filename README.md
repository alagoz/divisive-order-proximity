# Novel Divisive Hierarchical Clustering

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

Implementation of **Order-Based** and **Proximity-Based** Divisive Hierarchical Clustering algorithms, as proposed in our paper *"Order-Based and Proximity-Based Divisive Hierarchical Clustering for Improved Recursive Partitioning"*.

## 📋 Overview

This repository provides a modular implementation of two novel divisive hierarchical clustering algorithms:

- **Order-Based Divisive Clustering** - Splits clusters based on relative rank ordering of objects with respect to seed points
- **Proximity-Based Divisive Clustering** - Splits clusters using direct distance relationships with seed points

Both algorithms follow a top-down recursive partitioning strategy, offering interpretable hierarchical structures while maintaining competitive performance on real-world datasets.

### Key Features

- ✅ **Two Novel Algorithms** - Order-Based and Proximity-Based splitting strategies
- ✅ **Modular Design** - Easy to extend with new datasets and algorithms
- ✅ **Real-World Datasets** - Built-in support for 6 UCI datasets
- ✅ **Comprehensive Evaluation** - ARI, NMI, Silhouette metrics included
- ✅ **Visualization** - Side-by-side comparison of clustering results
- ✅ **Extensible** - Add your own datasets or splitting strategies

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/novel-divisive-clustering.git
cd novel-divisive-clustering

# Install dependencies
pip install -r requirements.txt

# Install the package (optional)
pip install -e .
```

## Run Demo
```bash
# Run with default dataset (Banknote Authentication)
python demo.py

# Specify a dataset
python demo.py --dataset credit_card

# Use cosine distance for high-dimensional data
python demo.py --dataset spambase --metric cosine

# Control number of clusters
python demo.py --dataset blobs --n_clusters 4

# Limit sample size for large datasets
python demo.py --dataset census_income --max_samples 2000
```

## Available Datasets
Dataset	Samples	Features	Classes	Domain
banknote	1,372	4	2	Finance
credit_card	30,000	23	2	Finance
credit_approval	690	15	2	Finance
german_credit	1,000	20	2	Finance
census_income	48,842	14	2	Social Science
spambase	4,601	57	2	Computer Science

## Example Usage in Python
```bash
from clustering import DivisiveTree, order_based_split, proximity_based_split
from datasets import load_banknote
from evaluation import evaluate_clustering, plot_results

# Load data
X, y, n_clusters, name = load_banknote()

# Run Order-Based clustering
tree = DivisiveTree(order_based_split, metric='euclidean')
pred_order = tree.fit(X, desired_clusters=n_clusters)

# Evaluate
metrics = evaluate_clustering(y, pred_order, X)
print(f"ARI: {metrics['ARI']:.4f}")
print(f"NMI: {metrics['NMI']:.4f}")
print(f"Silhouette: {metrics['Silhouette']:.4f}")

# Visualize
plot_results(X, y, pred_order, proximity_based_split, X, name)
```
## 📊 Results
Performance Comparison on Banknote Dataset
Algorithm	ARI	NMI	Silhouette	Time (s)
Order-Based	0.5684	0.3742	0.1845	0.0432
Proximity-Based	0.5512	0.3581	0.1798	0.0398
K-Means	0.5621	0.3654	0.1812	0.0082

Visualization Output
The demo generates visualizations showing:

- 2×2 Grid - Ground Truth, Order-Based, Proximity-Based, and K-Means
<img width="1400" height="1030" alt="Figure_clustering" src="https://github.com/user-attachments/assets/20c72413-f12e-4d17-ad42-5429bc902a04" />

- Metrics Comparison - Bar charts for ARI, NMI, Silhouette, and Runtime
<img width="1400" height="500" alt="Figure_metrics" src="https://github.com/user-attachments/assets/d15cafc4-ead9-4ad6-9c99-3735190b3df5" />

## 📁 Repository Structure
novel-divisive-clustering/  
├── clustering/              # Clustering algorithms  
│   ├── __init__.py  
│   ├── base.py             # DivisiveTree base class  
│   ├── order_based.py      # Order-Based splitter  
│   └── proximity_based.py  # Proximity-Based splitter  
├── datasets/               # Dataset loading  
│   ├── __init__.py  
│   ├── loader.py           # Dataset registry  
│   ├── banknote.py  
│   ├── credit_card.py  
│   ├── credit_approval.py  
│   ├── german_credit.py  
│   ├── census_income.py  
│   └── spambase.py  
├── evaluation/             # Evaluation & visualization  
│   ├── __init__.py  
│   ├── metrics.py          # ARI, NMI, Silhouette  
│   └── visualization.py    # Plotting functions  
├── utils/                  # Utilities  
│   ├── __init__.py  
│   └── helpers.py          # Helper functions  
├── demo.py                 # Main demo script  
├── requirements.txt        # Dependencies  
├── setup.py               # Package installation  
└── README.md              # This file  


## 🧪 Algorithms
Order-Based Divisive Clustering
The Order-Based algorithm splits clusters using relative rank ordering:

1. Select two maximally distant seed points

2. Rank all other points relative to each seed

3. Assign points based on which seed gives them a better rank

4. Use distance-based tie-breaking when ranks are equal

Key advantage: Robust to scale variations and outliers through rank-based comparison.  


Proximity-Based Divisive Clustering
The Proximity-Based algorithm uses direct distance relationships:

1. Select two maximally distant seed points

2. Assign each point to its nearest seed

3. Use rank-based tie-breaking when distances are equal

Key advantage: Intuitive partitions that preserve local geometric structure.


## 📈 Evaluation Metrics
The following metrics are included for evaluation:

- Adjusted Rand Index (ARI) - Measures similarity between predicted and true labels

- Normalized Mutual Information (NMI) - Measures mutual information between clusterings

- Silhouette Score - Measures intra-cluster cohesion vs. inter-cluster separation

- Runtime - Execution time in seconds

## 🔧 Adding New Datasets
To add a new dataset:

1. Create a new file in datasets/ (e.g., my_dataset.py)

2. Implement a loader function:
```bash
# datasets/my_dataset.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_my_dataset():
    """Load your dataset."""
    # Load data
    df = pd.read_csv('path/to/data.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Standardize
    X = StandardScaler().fit_transform(X)
    n_clusters = len(np.unique(y))
    
    return X, y, n_clusters, 'My Dataset (UCI)'
```

3. Register in ```datasets/loader.py```:
```bash
from .my_dataset import load_my_dataset

DATASETS = {
    # ...
    'my_dataset': load_my_dataset
}
```

## 🔬 Adding New Splitting Strategies
To implement a custom splitting strategy:
```bash
# clustering/my_splitter.py
import numpy as np

def my_custom_split(dist_matrix):
    """
    Custom splitting strategy.
    
    Parameters
    ----------
    dist_matrix : ndarray
        Pairwise distance matrix
        
    Returns
    -------
    ndarray : Binary labels
    """
    n = dist_matrix.shape[0]
    # Your splitting logic here
    labels = np.zeros(n, dtype=int)
    # ...
    return labels
```

## 📚 Citation
The work is under submission process.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository

2. Create your feature branch (git checkout -b feature/AmazingFeature)

3. Commit your changes (git commit -m 'Add some AmazingFeature')

4. Push to the branch (git push origin feature/AmazingFeature)

5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
UCI Machine Learning Repository for providing the datasets

Scikit-learn for the base clustering implementations

## 📧 Contact
For questions or feedback, please open an issue or contact:

Celal Alagöz - celal.alagoz@gmail.com

GitHub: @alagoz
    
