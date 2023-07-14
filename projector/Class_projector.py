#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import spatial, interpolate
from scipy.stats import multivariate_normal
from scipy.interpolate import LSQUnivariateSpline
from scipy.interpolate import BSpline, make_interp_spline
from sklearn.model_selection import train_test_split
import pandas as pd


# In[2]:


import skfda
from skfda.preprocessing.dim_reduction.projection import FPCA
from skfda.representation.basis import BSpline, FDataBasis
from sklearn.model_selection import cross_val_score
from skfda.ml.regression import KNeighborsRegressor


# In[3]:


from skfda.datasets import fetch_growth
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)


# In[4]:


import scipy.interpolate as spi
import pywt
import scipy.fftpack as spfft
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


# In[5]:


class OrthogonalExpansion:
    def __init__(self, n_basis=20, basis_type='identity'):
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.orthogonal_basis = None
        self.coefficients = None

    def gram_schmidt(self, vectors):
        """Gram-Schmidt orthogonalization process."""
        basis = []
        for v in vectors:
            w = v - sum(np.dot(v, b) * b for b in basis)
            if (w > 1e-10).any():
                basis.append(w / np.linalg.norm(w))
        return np.array(basis)

    def generate_initial_basis(self, data_shape):
        n_samples, n_features = data_shape

        if self.basis_type == 'identity':
            return np.eye(n_features)[:self.n_basis]
        elif self.basis_type == 'random':
            return np.random.randn(self.n_basis, n_features)
        elif self.basis_type == 'bspline':
            degree = 3
            x = np.linspace(0, 1, n_features)
            n_internal_knots = self.n_basis - (degree + 1)
            internal_knots = np.linspace(0, 1, n_internal_knots + 2)[1:-1]
            knots = np.concatenate(([0] * (degree + 1), internal_knots, [1] * (degree + 1)))
            coeffs = np.eye(self.n_basis)
            return spi.BSpline(knots, coeffs, degree, extrapolate=False)(x)
        elif self.basis_type == 'wavelet':
            wavelet = pywt.Wavelet('db4')  # Choose the desired wavelet type
            max_level = pywt.dwt_max_level(n_features, wavelet.dec_len)
            wpt = pywt.WaveletPacket(np.eye(n_features)[:self.n_basis], wavelet, mode='symmetric', maxlevel=max_level)
            wavelet_basis = []
            for node in wpt.get_leaf_nodes(True):
                coeffs = [np.zeros_like(node.data) if i != node.level - 1 else node.data for i in range(max_level)]
                basis_function = pywt.waverec(coeffs, wavelet)
                # Pad basis_function with zeros to match the data matrix's dimensions
                basis_function = np.pad(basis_function, (0, n_features - len(basis_function)), 'constant', constant_values=0)
                wavelet_basis.append(basis_function)
            return np.vstack(wavelet_basis)[:self.n_basis]
        elif self.basis_type == 'fourier':
            time_samples = np.linspace(0, 1, n_features)
            frequencies = np.fft.fftfreq(n_features, d=time_samples[1] - time_samples[0])
            return spfft.idct(np.eye(n_features)[:self.n_basis], axis=1, norm='ortho')
        else:
            raise ValueError("Invalid basis type. Supported types: 'identity', 'random', 'bspline', 'wavelet', 'fourier'")

    def fit(self, data_matrix):
        # Subtract the mean from the data
        mean_data_matrix = np.mean(data_matrix, axis=0)
        centered_data_matrix = data_matrix - mean_data_matrix

        # Create the initial basis vectors
        initial_basis = self.generate_initial_basis(centered_data_matrix.shape)

        # Perform the Gram-Schmidt process on the initial basis vectors
        self.orthogonal_basis = self.gram_schmidt(initial_basis)

        # Compute the coefficients a_iv by projecting the centered data onto the orthogonal basis
        self.coefficients = centered_data_matrix @ self.orthogonal_basis.T


# In[6]:


class projector:
    """
    A class that performs projection of an FDataGrid object using a specified
                        projection method and basis.

    Attributes:
        fdata (FDataGrid): The input functional data to project.
        projection_method (str): The projection method to use (e.g., 'fpca', 
                        'pca', 'kpca', 'expansion(orthogonalexpansion)' etc.).
        basis (str): The basis function to use for the projection (e.g., 'bspline', 
                        'wavelet', 'kernel', etc.).
        basis_args (dictionary): Additional argument for the projection method.
                Example:
                    B-spline basis: {'degree': 3, 'n_basis': 20}
                    Wavelet basis: {'wavelet': 'db4', 'n_basis': 20}
                    Kernel (RBF) basis: {'gamma': 1.0}
                    Fourier basis: {'n_basis': 20}

    Returns:
        tuple: A tuple containing the projection coefficients, mean, and basis functions.

    """

    def __init__(self, projection_method, basis, basis_args):
        """
        Initializes an FDataGridProjector object.
        
        Args:
            projection_method (str): The projection method to use (e.g., 'fpca', 'pca', 'kpca', etc.).
            basis (str): The basis function to use for the projection (e.g., 'bspline', '
                        wavelet', 'kernel', etc.).
            basis_args: Additional arguments for the projection method.
        """
        self.projection_method = projection_method
        self.basis = basis
        self.basis_args = basis_args
        self.orthogonal_basis = None
        self.coefficients = None
        
    def gram_schmidt(self, vectors):
        """Gram-Schmidt orthogonalization process."""
        basis = []
        for v in vectors:
            w = v - sum(np.dot(v, b) * b for b in basis)
            if (w > 1e-10).any():
                basis.append(w / np.linalg.norm(w))
        return np.array(basis)
    
    def is_orthogonal(self, matrix):
        """Judge orthogonalization."""
        transpose = np.transpose(matrix)
        product = np.dot(matrix, transpose)
        identity = np.identity(matrix.shape[0])
        return np.allclose(product, identity)
    
    def generate_initial_basis(self, data_shape):
        """Generate initial basis for orthogonal expansion"""
        n_samples, n_features = data_shape

        if self.basis == 'identity':
            return np.eye(n_features)[:self.n_basis]
        elif self.basis == 'random':
            return np.random.randn(self.n_basis, n_features)
        
        elif self.basis == 'bspline':
            degree = self.basis_args.get('degree', 3)
            n_basis = self.basis_args.get('n_basis', n_features)

            x = np.linspace(0, 1, n_features)
            n_internal_knots = n_basis - (degree + 1)
            internal_knots = np.linspace(0, 1, n_internal_knots + 2)[1:-1]
            knots = np.concatenate(([0] * (degree + 1), internal_knots, [1] * (degree + 1)))
            coeffs = np.eye(n_basis)

            return spi.BSpline(knots, coeffs, degree, extrapolate=False)(x)
        
        elif self.basis == 'wavelet':
            wavelet_name = self.basis_args.get('wavelet', 'db4')
            n_basis = self.basis_args.get('n_basis', n_features)

            wavelet = pywt.Wavelet(wavelet_name)
            max_level = pywt.dwt_max_level(n_features, wavelet.dec_len)
            wpt = pywt.WaveletPacket(np.eye(n_features)[:n_basis], wavelet, mode='symmetric', maxlevel=max_level)

            wavelet_basis = []
            for node in wpt.get_leaf_nodes(True):
                coeffs = [np.zeros_like(node.data) if i != node.level - 1 else node.data for i in range(max_level)]
                basis_function = pywt.waverec(coeffs, wavelet)
                basis_function = np.pad(basis_function, (0, n_features - len(basis_function)), 'constant', constant_values=0)
                wavelet_basis.append(basis_function)

            return np.vstack(wavelet_basis)[:n_basis]
            
            
        elif self.basis == 'fourier':
            n_basis = self.basis_args.get('n_basis', n_features)

            x = np.linspace(0, 1, n_features)
            freqs = np.fft.fftfreq(n_basis, d=1/n_features)
            fourier_basis = np.zeros((n_basis, n_features))

            for i in range(n_basis):
                temp = np.zeros(n_basis)
                temp[i] = 1
                fourier_basis[i] = spfft.idct(temp, norm='ortho')

            return fourier_basis

        
        else:
            raise ValueError("Invalid basis type. Supported types: 'identity', 'random', 'bspline', 'wavelet', 'fourier'")
        

    def fit_orthogonal_expansion(self, data_matrix):
        # Subtract the mean from the data
        mean_data_matrix = np.mean(data_matrix, axis=0)
        centered_data_matrix = data_matrix - mean_data_matrix

        # Create the initial basis vectors
        initial_basis = self.generate_initial_basis(centered_data_matrix.shape)

        # Perform the Gram-Schmidt process on the initial basis vectors if basis is not orthogonal
        if self.is_orthogonal(initial_basis) == False:
#             print(self.is_orthogonal(initial_basis))
            self.orthogonal_basis = self.gram_schmidt(initial_basis)
            
        elif self.is_orthogonal(initial_basis) == True:
#             print(self.is_orthogonal(initial_basis))
            self.orthogonal_basis = np.array(initial_basis)

        # Compute the coefficients a_iv by projecting the centered data onto the orthogonal basis
        self.coefficients = centered_data_matrix @ self.orthogonal_basis.T
        
        return self.coefficients
    
    
    def fit_transform(self, fdata):
        """
        Fits the projection model to the input data and transforms the input data.
        
        Args:
            fdata (FDataGrid): The input functional data to project.
        
        Returns:
            tuple: A tuple containing the projection coefficients, mean, and basis functions.
        """
        
        n_samples, n_features = fdata.data_matrix.reshape(fdata.n_samples, -1).shape
        
        if self.projection_method == 'expansion' :
            data_matrix = fdata.data_matrix.reshape(fdata.n_samples, -1)
            projection_coefficients = self.fit_orthogonal_expansion(data_matrix)
            
            return projection_coefficients
        
        # Create the basis object based on the specified basis function
        if self.basis is not None:
            if self.basis == 'bspline':
                degree = self.basis_args.get('degree', 3)
                n_basis = self.basis_args.get('n_basis', n_features)
                
                basis = skfda.representation.basis.BSplineBasis(n_basis=n_basis, order=degree)
                smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
                fd_smooth = smoother.fit_transform(fd)
            elif self.basis == 'fourier':
                n_basis = self.basis_args.get('n_basis', n_features)
                period = self.basis_args.get('period', 1)
                
                basis = skfda.representation.basis.FourierBasis((0, 1), n_basis=n_basis, period = period)
                smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
                fd_smooth = smoother.fit_transform(fd)
            elif self.basis == 'kernel':
                bandwidth = self.basis_args.get('bandwidth', 3.5)
                
                kernel_estimator = NadarayaWatsonHatMatrix(bandwidth=bandwidth)
                smoother = KernelSmoother(kernel_estimator=kernel_estimator)
                fd_smooth = smoother.fit_transform(fd)
            else:
                raise ValueError(f"Invalid smoother type: {self.smoother}")
        
        n_components = self.basis_args.get('n_components', 4)
        
        # Create the projection object based on the specified projection method
        if self.projection_method == 'fpca':
            projection_obj = skfda.preprocessing.dim_reduction.projection.FPCA(n_components=n_components)
        elif self.projection_method == 'pca':
            projection_obj = skfda.preprocessing.dim_reduction.projection.PCA(basis=basis)
        elif self.projection_method == 'kpca':
            projection_obj = skfda.preprocessing.dim_reduction.projection.KPCA(basis=basis)
        else:
            raise ValueError(f"Invalid projection method: {self.projection_method}")
            
        # Fit the projection object to the input functional data
        projection_obj.fit(fdata)
        
        # Get the projection coefficients, mean, and basis functions
        projection_coefficients = projection_obj.transform(fdata)
#         mean = projection_obj.mean_.data_matrix[0]
#         basis_functions = basis_obj.evaluate(projection_obj.domain_range[0])
        
        self.coefficients = projection_coefficients
        
        # Return the projection coefficients, mean, and basis functions
        return projection_coefficients


# In[7]:


dataset = skfda.datasets.fetch_growth()
fd = dataset['data']
y = dataset['target']
# fd.plot()


# In[8]:


tem = projector(projection_method = 'expansion', basis = 'bspline', basis_args = {'degree': 3, 'n_basis': 31})
coefficients = tem.fit_transform(fd)
coefficients


# In[9]:


tem = projector(projection_method = 'expansion', basis = 'fourier', basis_args = {'n_basis': 31})
coefficients = tem.fit_transform(fd)
coefficients


# In[10]:


tem = projector(projection_method = 'expansion', basis = 'wavelet', basis_args = {'wavelet': 'db4', 'n_basis': 20})
coefficients = tem.fit_transform(fd)
coefficients


# In[ ]:




