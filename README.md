# GPmixture
---
### Introduction

GPmixture is a Python package for model-based clustering on Gaussian mixture models.

---
### Initialisation

First, you need to load some dependent libraries.

```python
import numpy as np
import skfda
import pywt
import scipy
import sklearn
from Class_smoother import smoother
from Class_projector import projector
from Class_gmm import fdata_gmm
```

---
### Smoother

Smoother is a class that can smooth raw data. It's a first step in this package.

In order to use Smoother, that is, the data is smoothed by conversion and converted into a more suitable data type FDataGrid. FdataGrid is a data type from the skfda library, which is more convenient for the following operations.

```python
tem = smoother(smoother = 'bspline', smoother_args = {'degree': 3, 'n_basis': 31}, domain_range = (10))
fdata_smooth = tem.fit(X)
```

- smoother (str): The smoother to use (e.g., 'bspline', 'fourier', 'kernel', 'wavelet').
- smoother_args (dict): Additional arguments for the smoother (e.g., {'bandwidth': 0.1} for a kernel smoother).
- domain_range (tuple): The domain range of the functional data.
- X (ndarray): The input data to transform.

---
### Projector

Projecter is a class that performs the projection of an FDataGrid object using a specified projection method and basis. It's the second step in this package.

```python
tem = projector(projection_method = 'expansion', basis = 'bspline', basis_args = {'degree': 3, 'n_basis': 31})
coefficients = tem.fit_transform(fd)
```

- projection_method (str): The projection method to use (e.g., 'fpca', 'pca', 'kpca', 'expansion'.).
- basis (str): The basis function to use for the projection (e.g., 'bspline', 'wavelet', 'kernel', 'fourier'.).
- basis_args: Additional arguments for the projection method.

---
### GMM

GMM is a class that performs clustering on projection coefficients using Gaussian Mixture Models (GMMs). It's the third step in this package.

```python
gmm = fdata_gmm(3, 'full')
gmms = gmm.fit(X)
gmm.membership_matrix()
gmm.binary_membership_matrix()
gmm.calculate_weights()
```

- projection_coefficients (ndarray): The input data in the form of projection coefficients.
- n_components (int): The number of mixture components to use in the GMM.
- covariance_type (str): The type of covariance matrix to use in the GMM (e.g., 'full', 'tied', 'diag', 'spherical').

Then the next attributes returned are the results obtained through GMM.
- membership_indicator_matrix (ndarray): A binary membership indicator matrix that indicates the cluster membership of each data point.
- cluster_membership (ndarray): An array of cluster membership values that indicates the cluster membership of each data point.
- calculate_weights (ndarray): An array of final weights that can be used directly through the clustering method.

---
### Usage

The complete process of using this library is as follows：

```python
from Class_smoother import smoother
Smoother = smoother(smoother = 'bspline', smoother_args = {'degree': 3, 'n_basis': 10})
fdata_smooth = Smoother.fit(Y)

from Class_projector import projector
Projector = projector(projection_method = 'fpca', basis = 'bspline', basis_args = {'degree': 3, 'n_basis': 5, 'n_components': 4})
coefficients = Projector.fit_transform(fdata_smooth)

from Class_gmm import fdata_gmm
gmm = fdata_gmm(n_components = 4)
gmms = gmm.fit(coefficients)
membership_matrix = gmm.membership_matrix()
affinity_matrix = gmm.affinity_matrix()
weights = gmm.calculate_weights()

from sklearn.cluster import SpectralClustering
sc = SpectralClustering(n_clusters=4, affinity='precomputed', assign_labels='discretize')
sc.fit(weights)
clustering_label = sc.labels_

# Get the AMI score
from sklearn.metrics.cluster import adjusted_mutual_info_score
print("AMI score:")
print(adjusted_mutual_info_score(label, clustering_label))

# Get the ARM score
from sklearn.metrics.cluster import adjusted_rand_score
print("ARM score:")
print(adjusted_rand_score(label, clustering_label))
```
