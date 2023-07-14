#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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
        
        # Smooth the input data if a smoother is specified
        if self.smoother is not None:
            if self.smoother == 'bspline':
                degree = self.smoother_args.get('degree', 3)
                n_basis = self.smoother_args.get('n_basis', n_features)
                
                basis = skfda.representation.basis.BSplineBasis(n_basis=n_basis, order=degree)
                smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
                fd_smooth = smoother.fit_transform(fd)
            elif self.smoother == 'fourier':
                n_basis = self.smoother_args.get('n_basis', n_features)
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
                wavelet_type = self.smoother_args.get('wavelet', 'db2')
                level = 2

                # Perform the wavelet decomposition
                coeffs = pywt.wavedec2(data, wavelet_type, level=level)

                # Define a thresholding function
                def threshold_coeffs(coeffs, threshold):
                    thresholded_coeffs = []
                    for i, coeff in enumerate(coeffs):
                        if i == 0:
                            thresholded_coeffs.append(coeff)
                        else:
                            thresholded_coeffs.append(tuple(pywt.threshold(c, threshold, mode='soft') for c in coeff))
                    return thresholded_coeffs

                # Set the threshold value and apply the thresholding function
                threshold_value = 1
                thresholded_coeffs = threshold_coeffs(coeffs, threshold_value)

                # Reconstruct the smoothed data
                smoothed_data = pywt.waverec2(thresholded_coeffs, wavelet_type)


                # Set the threshold value and apply the thresholding function
                threshold_value = 1
                thresholded_coeffs = threshold_coeffs(coeffs, threshold_value)

                # Reconstruct the smoothed data
                smoothed_data = pywt.waverec2(thresholded_coeffs, wavelet_type)
                fd_smooth = skfda.FDataGrid(data_matrix=smoothed_data)
                

            else:
                raise ValueError(f"Invalid smoother type: {self.smoother}")
                
        
        
        # Return the transformed functional data
        return fd_smooth


# In[3]:


dataset = skfda.datasets.fetch_growth()
fd = dataset['data']
y = dataset['target']


# In[4]:


X = fd.data_matrix.reshape(fd.n_samples, -1)


# In[5]:


tem = smoother(smoother = 'bspline', smoother_args = {'degree': 3, 'n_basis': 31})
fdate_smooth = tem.fit(X)
fdate_smooth


# In[6]:


tem = smoother(smoother = 'fourier', smoother_args = {'n_basis': 10, 'period': 1})
fdate_smooth = tem.fit(X)
fdate_smooth


# In[7]:


tem = smoother(smoother = 'kernel', smoother_args = {'bandwidth': 0.1})
fdate_smooth = tem.fit(X)
fdate_smooth


# In[8]:


tem = smoother(smoother = 'wavelet', smoother_args = {'wavelet': 'db4'})
fdate_smooth = tem.fit(X)
fdate_smooth


# In[9]:


data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
grid_points = [[2, 4], [3,6]]
fd = FDataGrid(data_matrix)


# In[10]:


data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Set the wavelet type and decomposition level
wavelet_type = 'db2'
level = 1

# Perform the wavelet decomposition
coeffs = pywt.wavedec2(data, wavelet_type, level=level)

# Define a thresholding function
def threshold_coeffs(coeffs, threshold):
    thresholded_coeffs = []
    for i, coeff in enumerate(coeffs):
        if i == 0:
            thresholded_coeffs.append(coeff)
        else:
            thresholded_coeffs.append(tuple(pywt.threshold(c, threshold, mode='soft') for c in coeff))
    return thresholded_coeffs

# Set the threshold value and apply the thresholding function
threshold_value = 1
thresholded_coeffs = threshold_coeffs(coeffs, threshold_value)

# Reconstruct the smoothed data
smoothed_data = pywt.waverec2(thresholded_coeffs, wavelet_type)



# In[ ]:




