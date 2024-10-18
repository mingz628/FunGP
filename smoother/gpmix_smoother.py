#!/usr/bin/env python
# coding: utf-8

import numpy as np
import skfda
from skfda import FDataGrid
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
from skfda.preprocessing.smoothing import KernelSmoother, BasisSmoother
import scipy.interpolate as spi
import pywt
import scipy.fftpack as spfft
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler



class smoother:
    """
    A class that transforms a numpy ndarray to an FDataGrid object.

    Attributes:
        smoother (str): The smoother to use (e.g., 'bspline', 'fourier', 'kernel').
        smoother_args (dict): Additional arguments for the smoother (e.g., {'bandwidth': 0.1} for a kernel smoother).
        domain_range (tuple): The domain range of the functional data.
        X (ndarray): The input data to transform.

    Returns:
        fdata (FDataGrid): the transformed and smoothed data.

    """
    
    def __init__(self, smoother='spline', smoother_args=None, domain_range=None):
        """
        Initializes an NdarrayToFDataGrid object.
        
        Args:
            smoother (str): The smoother to use (e.g., 'spline', 'savitzky_golay', 'kernel').
            smoother_args (dict): Additional arguments for the smoother (e.g., {'bandwidth': 0.1} for a kernel smoother).
            domain_range (tuple): The domain range of the functional data.
            
            Example:
                    B-spline basis: {'degree': 3, 'n_basis': 20}
                    Wavelet basis: {'wavelet': 'db4', 'n_basis': 20}
                    Kernel basis: {'bandwidth': 1.0}
                    Fourier basis: {'n_basis': 20, 'period': 1}
        """
        self.smoother = smoother
        self.smoother_args = smoother_args
        self.domain_range = domain_range
        
    def fit(self, X):
        """
        Fits the transformation and transforms the input data.
        
        Args:
            X (array-like): The input data to transform.
        
        Returns:
            FDataGrid: The transformed functional data.
        """
        # Get the dimensions of the input data
        n_samples, n_features = X.shape
        
        # Create the domain of the functional data
        if self.domain_range is None:
            domain_range = np.linspace(0, 1, n_features)
        else:
            domain_range = self.domain_range
            
        
        fd = skfda.FDataGrid(data_matrix=X, grid_points=domain_range)
        self.data = fd
        
        # Smooth the input data if a smoother is specified
        if self.smoother is not None:
            if self.smoother == 'bspline':
                degree = self.smoother_args.get('degree', 3)
                n_basis = self.smoother_args.get('n_basis', 10)
                
                basis = skfda.representation.basis.BSplineBasis(n_basis=n_basis, order=degree)
                smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
                fd_smooth = smoother.fit_transform(fd)
            elif self.smoother == 'fourier':
                n_basis = self.smoother_args.get('n_basis', 10)
                period = self.smoother_args.get('period', 1)
                
                basis = skfda.representation.basis.FourierBasis((0, 1), n_basis=n_basis, period = period)
                smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
                fd_smooth = smoother.fit_transform(fd)
            elif self.smoother == 'kernel':
                bandwidth = self.smoother_args.get('bandwidth', 3.5)
                
                kernel_estimator = NadarayaWatsonHatMatrix(bandwidth=bandwidth)
                smoother = KernelSmoother(kernel_estimator=kernel_estimator)
                fd_smooth = smoother.fit_transform(fd)
                
            elif self.smoother == 'wavelet':
                data = X

                # Set the wavelet type and decomposition level
                wavelet = self.smoother_args.get('wavelet', 'db2')
                level = self.smoother_args.get('level', 2)

                # Perform the wavelet decomposition
                coeffs = pywt.wavedec(data, wavelet, level=level)
                threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(data)))
                coeffs[1:] = (pywt.threshold(c, threshold, 'soft') for c in coeffs[1:])
                smoothed_data = pywt.waverec(coeffs, wavelet)
#                 print(smoothed_data.shape)
                


                # Reconstruct the smoothed data
#                 fd_smooth = skfda.FDataGrid(data_matrix=smoothed_data, grid_points = domain_range)
                fd_smooth = skfda.FDataGrid(data_matrix=smoothed_data)
                self.fd_smooth = fd_smooth

            else:
                raise ValueError(f"Invalid smoother type: {self.smoother}")
                
            self.fd_smooth = fd_smooth
        if self.smoother != 'wavelet':
            self.fpa_smoother = smoother
        else:
            basis = skfda.representation.basis.BSplineBasis(n_basis=30, order=10)
            self.fpa_smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
            
        
        # Return the transformed functional data
        return fd_smooth
    
    def plot(self):
        self.fd_smooth.plot()
        
    def get_score(self, X, y):
        return self.fpa_smoother.score(X = X, y = y)
    
    def score(self):
        return self.fpa_smoother.score(X = self.data, y = self.data)