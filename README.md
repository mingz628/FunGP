# GPmixture
---
### Introduction

GPmixture is an ensemble clustering framework that can efficiently identify the latent cluster labels of functional data from a Gaussian process mixture.

---
### Initialisation

To get started with GPmixture, make sure to import and install the necessary libraries:

```python
import numpy as np
import skfda
import pywt
import scipy
import sklearn
from smoother.gpmix_smoother import smoother
from projector.gpmix_projector import projector
from unigmm.gpmix_gmm import unigmm
```

---
### Smoother

The smoother class is used to smooth raw data, which is the first step in this package. The data is smoothed and converted into a more suitable data type called FDataGrid. FDataGrid is a data type from the skfda library, which is more convenient for subsequent operations.

To use the smoother class, follow these steps:

```python
gpmix_smoother = smoother(smoother = 'bspline', smoother_args = {'degree': 3, 'n_basis': 31}, domain_range = (10))
fdata_smoothed = gpmix_smoother.fit(X)
```

- smoother (str): The smoother to use (e.g., 'bspline', 'fourier', 'kernel', 'wavelet').
- smoother_args (dict): Additional arguments for the smoother (e.g., {'bandwidth': 0.1} for a kernel smoother).
- domain_range (tuple): The domain range of the functional data.
- X (ndarray): The input data to transform.

---
### Projector

The projector class performs the projection of an FDataGrid object using a specified projection method and basis. It is the second step in this package.

Here's an example of using the projector class:

```python
gpmix_projector = projector(projection_method = 'expansion', projection_args = {'basis': 'bspline', 'degree': 3, 'n_basis': 31})
projection_coefficients = gpmix_projector.fit(fdata_smoothed)
```

- projection_method (str): The projection method to use (e.g., 'fpca', 'pca', 'kpca', 'expansion'.).
- projection_args: Additional arguments for the projection methode(e.g., 'basis': 'bspline', 'wavelet', 'kernel', 'fourier' and arguments).

---
### Unigmm

The unigmm class performs clustering on projection coefficients using Gaussian Mixture Models (GMMs). It is the third step in this package.

```python
gpmix_unigmm = unigmm(3, 'full')
gpmix_unigmm.fit(projection_coefficients)
gpmix_unigmm.membership_matrix()
gpmix_unigmm.binary_membership_matrix()
gpmix_unigmm.calculate_weights()
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

To use the GPmixture library, follow the complete process outlined below:

```python
from smoother.gpmix_smoother import smoother
gpmix_smoother = smoother(smoother = 'bspline', smoother_args = {'degree': 3, 'n_basis': 10})
fdata_smoothed = gpmix_smoother.fit(Y)

from projector.gpmix_projector import projector
gpmix_projector = projector(projection_method = 'expansion', projection_args = {'basis': 'bspline', 'degree': 3, 'n_basis': 31})
coefficients = Projector.fit(fdata_smoothed)

from unigmm.gpmix_unigmm import unigmm
gpmix_unigmm = unigmm(n_components = 4)
gpmix_unigmm.fit(coefficients)
membership_matrix = gpmix_unigmm.membership_matrix()
affinity_matrix = gpmix_unigmm.affinity_matrix()
weights = gpmix_unigmm.calculate_weights()

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
