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


import numpy as np
import scipy.stats as stats
from scipy.integrate import quad


# In[4]:


class fdata_gmm:
    """
    A class that performs clustering on projection coefficients using Gaussian Mixture Models (GMMs).

    Attributes:
        projection_coefficients (ndarray): The input data in the form of projection coefficients.
        n_components (int): The number of mixture components to use in the GMM.
        covariance_type (str): The type of covariance matrix to use in the GMM (e.g., 'full', 'tied', 'diag', 'spherical').
        clustering_method (str): The clustering method to apply after fitting the GMM (e.g., 'kmeans').

    Returns:
        membership_indicator_matrix (ndarray): A binary membership indicator matrix that indicates the cluster membership of each data point.
        cluster_membership (ndarray): An array of cluster membership values that indicates the cluster membership of each data point.
    """
    def __init__(self, n_components, covariance_type='full', clustering_method='kmeans',  random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.clustering_method = clustering_method
        self.random_state = random_state
        self.gmms = []

    def fit(self, X):
        """
        Fit the Gaussian Mixture Model for the data.

        Returns: 
            A list of fitted GMMs for each row of the data.
        """
        self.projection_coefficients = X
        gmms = []
        for row in self.projection_coefficients:
            gmm = GaussianMixture(
                n_components=self.n_components, 
                covariance_type=self.covariance_type,
                random_state=self.random_state
            )
            gmm.fit(row.reshape(-1, 1))
            gmms.append(gmm)
            
        self.gmms = gmms
        return gmms
    
    def predict(self, samples):
        """
        Predicts the component labels for each sample in samples using the Gaussian Mixture Models.

        """
        predictions = []
        for model, sample in zip(self.models, samples):
            pred = model.predict(sample.reshape(1, -1))
            predictions.append(pred[0])
        return np.array(predictions)

    def membership_matrix(self):
        """
        Compute the cluster membership matrix using the fitted GMMs.

        Returns:  
            (ndarray) The cluster membership matrix.
        """
        membership_matrix = np.array([gmm.predict_proba(row.reshape(-1, 1)) for row, gmm in zip(self.projection_coefficients, self.gmms)])
        self.membership_matrix = membership_matrix
        return membership_matrix

    def binary_membership_matrix(self):
        """
        Construct a binary membership indicator matrix from the cluster membership matrix.

        Returns:  
            (ndarray) The binary membership matrix.
        """
        membership_matrix = self.membership_matrix
        
        binary_matrix = np.zeros_like(membership_matrix)
        max_indices = np.argmax(membership_matrix, axis=2)
        x_indices, y_indices = np.meshgrid(np.arange(membership_matrix.shape[0]), np.arange(membership_matrix.shape[1]), indexing='ij')
        binary_matrix[x_indices, y_indices, max_indices] = 1
        
        self.binary_matrix = binary_matrix
        return binary_matrix

    def calculate_weights(self):
        """
        Calculate the weights for each column of data.

        Returns: 
            (ndarray) The weight matrix.
        """
        
        n_components = self.n_components
        
        membership_matrices = []
        n = len(self.projection_coefficients[0])
        for i in range(self.n_components):
            gmm = GaussianMixture(n_components=n_components, covariance_type='full')
            gmm.fit(self.projection_coefficients[:, i].reshape(-1, 1))
            labels = gmm.predict(self.projection_coefficients[:, i].reshape(-1, 1))
            #Create the membership probabilities for each component
            probability = gmm.predict_proba(self.projection_coefficients[:, i].reshape(-1,1))
            membership_matrices.append(probability)
        matrix_multiplication = []
        for i in range(self.n_components):
            matrix_multiplication.append(np.matmul(membership_matrices[i], membership_matrices[i].T))
        self.weights = sum(matrix_multiplication)
        return self.weights


    def affinity_matrix(self):
        membership_matrices = self.membership_matrix
        matrix_multiplication = []
        for i in range(self.n_components):
            matrix_multiplication.append(np.matmul(membership_matrices[i], membership_matrices[i].T))
        self.affinity_matrix = sum(matrix_multiplication)
        return self.affinity_matrix
        
        
# In[5]:


dataset = skfda.datasets.fetch_growth()
fd = dataset['data']
y = dataset['target']


# In[6]:


X = fd.data_matrix.reshape(fd.n_samples, -1)


# In[7]:


gmm = fdata_gmm(3, 'full')
gmms = gmm.fit(X)


# In[8]:


gmm.membership_matrix()


# In[9]:


gmm.binary_membership_matrix()


# In[10]:


def gaussian_pdf(x, mean, std_dev):
    return stats.norm.pdf(x, mean, std_dev)


# In[11]:


def misclassification_probability(mean_k, std_dev_k, pi_k, mean_r, std_dev_r, pi_r):
    def integrand(x):
        pdf_k = gaussian_pdf(x, mean_k, std_dev_k)
        pdf_r = gaussian_pdf(x, mean_r, std_dev_r)
        return pdf_k * pi_r / (pi_k * pdf_k + pi_r * pdf_r)
    
    w_rk, _ = quad(integrand, -np.inf, np.inf)
    return w_rk


# In[12]:


mean_k = 0
std_dev_k = 1
pi_k = 0.5

mean_r = 0
std_dev_r = 1
pi_r = 0.5

w_rk = misclassification_probability(mean_k, std_dev_k, pi_k, mean_r, std_dev_r, pi_r)
# print("Misclassification probability w_rk:", w_rk)


# In[13]:


# Define parameters for mixture components
p_i = 0.6
p_j = 0.4
mu_i = np.array([1, 2])
mu_j = np.array([4, 3])
Sigma_i = np.array([[2, 1], [1, 2]])
Sigma_j = np.array([[3, 1], [1, 3]])

# Define density functions for mixture components
def phi_i(x):
    return multivariate_normal.pdf(x, mean=mu_i, cov=Sigma_i)

def phi_j(x):
    return multivariate_normal.pdf(x, mean=mu_j, cov=Sigma_j)


# In[14]:


def w_i_given_j(x):
    numerator = p_i * phi_i(x)
    denominator = p_i * phi_i(x) + p_j * phi_j(x)
    return numerator / denominator


# In[15]:


x = np.array([2, 3])
w = w_i_given_j(x)
# print(f"for instance {x} : {w:.4f}")


