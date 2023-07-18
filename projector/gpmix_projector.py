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
        projection_args (dictionary): Additional argument for the projection method.
                Example:
                    B-spline basis: {'degree': 3, 'n_basis': 20}
                    Wavelet basis: {'wavelet': 'db4', 'n_basis': 20}
                    Kernel (RBF) basis: {'gamma': 1.0}
                    Fourier basis: {'n_basis': 20}

    Returns:
        tuple: A tuple containing the projection coefficients, mean, and basis functions.

    """

    def __init__(self, projection_method, projection_args):
        """
        Initializes an FDataGridProjector object.
        
        Args:
            projection_method (str): The projection method to use (e.g., 'expansion', 'fpca', 'pca', 'kpca', etc.).
            projection_args: Additional arguments for the projection method.
        """
        self.projection_method = projection_method
        self.projection_args = projection_args
        self.orthogonal_basis = None
        self.coefficients = None
        
        
    def plot(self):
        self.fdata.plot()
        
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
            n_basis = self.projection_args.get('n_basis', n_features)
            return np.eye(n_features)[:n_basis]
        elif self.basis == 'random':
            n_basis = self.projection_args.get('n_basis', n_features)
            return np.random.randn(n_basis, n_features)
        
        elif self.basis == 'bspline':
            degree = self.projection_args.get('degree', 3)
            n_basis = self.projection_args.get('n_basis', n_features)

            x = np.linspace(0, 1, n_features)
            n_internal_knots = n_basis - (degree + 1)
            internal_knots = np.linspace(0, 1, n_internal_knots + 2)[1:-1]
            knots = np.concatenate(([0] * (degree + 1), internal_knots, [1] * (degree + 1)))
            coeffs = np.eye(n_basis)

            return spi.BSpline(knots, coeffs, degree, extrapolate=False)(x)
        
        elif self.basis == 'wavelet':
            wavelet_name = self.projection_args.get('wavelet', 'db4')
            n_basis = self.projection_args.get('n_basis', n_features)

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
            n_basis = self.projection_args.get('n_basis', n_features)

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

#         print(centered_data_matrix.shape)
#         print(self.orthogonal_basis.shape)
        # Compute the coefficients a_iv by projecting the centered data onto the orthogonal basis
        self.coefficients = centered_data_matrix @ self.orthogonal_basis.T
        
        return self.coefficients
    
    def fit_fpca(self, fdata):
        
        if self.basis == None:
        
            n_components = self.projection_args.get('n_components', 3)
            fpca = skfda.preprocessing.dim_reduction.FPCA(n_components = n_components, centering = True)
            fpca.fit(fdata)
            self.coefficients = fpca.components_.data_matrix.reshape(fpca.components_.n_samples, -1)

            return self.coefficients
        
        elif self.basis == 'MonomialBasis':
            domain_range = self.projection_args.get('domain_range', (0,1))
            n_basis = self.projection_args.get('n_basis', 2)
            basis = skfda.representation.basis.MonomialBasis(
                domain_range=domain_range, n_basis=n_basis
            )
            basis_fd = fdata.to_basis(basis)
            fpca_basis = FPCA(2)
            fpca_basis = fpca_basis.fit(basis_fd)
            self.coefficients = fpca_basis.components_.data_matrix.reshape(fpca.components_.n_samples, -1)
            return self.coefficients
        
        
        
    def fit_pca(self, fdata):
        data_matrix = fdata.data_matrix.squeeze()
        mean_data_matrix = np.mean(data_matrix, axis=0)
        centered_data_matrix = data_matrix - mean_data_matrix
        n_components = self.projection_args.get('n_components', 3)
        pca = sklearn.decomposition.PCA(n_components = n_components)
        pca.fit(centered_data_matrix)
        self.coefficients = pca.components_
        
        return self.coefficients
        
    def fit_kpca(self, fdata):
        data_matrix = fdata.data_matrix.squeeze()
        mean_data_matrix = np.mean(data_matrix, axis=0)
        centered_data_matrix = data_matrix - mean_data_matrix
        n_components = self.projection_args.get('n_components', 3)
        kernel = self.projection_args.get('kernel', 'rbf')
        kpca = sklearn.decomposition.KernelPCA(n_components = n_components, kernel = kernel)
        kpca.fit(centered_data_matrix.T)
        self.coefficients = kpca.eigenvectors_
        
        return self.coefficients
        
    
    def fit(self, fdata):
        """
        Fits the projection model to the input data and transforms the input data.
        
        Args:
            fdata (FDataGrid): The input functional data to project.
        
        Returns:
            ndarray: A ndarray that is projection coefficients.
        """
        
        

        
        if self.projection_method == 'expansion' :
            self.basis = self.projection_args.get('basis', 'bspline')
            data_matrix = fdata.data_matrix.squeeze()
            projection_coefficients = self.fit_orthogonal_expansion(data_matrix.T)
            
#             print('expansion')
            
            return projection_coefficients
    
        elif self.projection_method == 'fpca':
            self.basis = self.projection_args.get('basis', None)
            projection_coefficients = self.fit_fpca(fdata)
            return projection_coefficients.T
        
        elif self.projection_method == 'kpca':
            projection_coefficients = self.fit_kpca(fdata)
            return projection_coefficients
        
        elif self.projection_method == 'pca':
            projection_coefficients = self.fit_pca(fdata)
            return projection_coefficients.T
        
        else:
            raise ValueError(f"Invalid projection method: {self.projection_method}")
        
#         # Create the basis object based on the specified basis function
#         if self.basis is not None:
#             if self.basis == 'bspline':
#                 degree = self.projection_args.get('degree', 3)
#                 n_basis = self.projection_args.get('n_basis', n_features)
                
#                 basis = skfda.representation.basis.BSplineBasis(n_basis=n_basis, order=degree)
#                 smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
#                 fd_smooth = smoother.fit_transform(fd)
#             elif self.basis == 'fourier':
#                 n_basis = self.projection_args.get('n_basis', n_features)
#                 period = self.projection_args.get('period', 1)
                
#                 basis = skfda.representation.basis.FourierBasis((0, 1), n_basis=n_basis, period = period)
#                 smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
#                 fd_smooth = smoother.fit_transform(fd)
#             elif self.basis == 'kernel':
#                 bandwidth = self.projection_args.get('bandwidth', 3.5)
                
#                 kernel_estimator = NadarayaWatsonHatMatrix(bandwidth=bandwidth)
#                 smoother = KernelSmoother(kernel_estimator=kernel_estimator)
#                 fd_smooth = smoother.fit_transform(fd)
#             else:
#                 raise ValueError(f"Invalid smoother type: {self.smoother}")
        
#         # Create the projection object based on the specified projection method
#         if self.projection_method == 'fpca':
#             projection_obj = skfda.decomposition.FPCA(basis=basis)
#         elif self.projection_method == 'pca':
#             projection_obj = skfda.decomposition.PCA(basis=basis)
#         elif self.projection_method == 'kpca':
#             projection_obj = skfda.kernel_pca.KPCA(basis=basis)
#         else:
#             raise ValueError(f"Invalid projection method: {self.projection_method}")
            
#         # Fit the projection object to the input functional data
#         projection_obj.fit(fdata)
        
#         # Get the projection coefficients, mean, and basis functions
#         projection_coefficients = projection_obj.transform(fdata).data_matrix[0]
#         mean = projection_obj.mean_.data_matrix[0]
#         basis_functions = basis_obj.evaluate(projection_obj.domain_range[0])
        
#         self.coefficients = projection_coefficients
        
#         # Return the projection coefficients, mean, and basis functions
#         return projection_coefficients