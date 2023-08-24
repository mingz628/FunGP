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



def _density_broad_search_star(a_b):
    try:
        return euclidean_distances(a_b[1], a_b[0])
    except Exception as e:
        raise Exception(e)


def _find_density_spherical_bandwidth(X, n_samples, n_features):
    center = X.sum(0) / n_samples
    X_centered = X - center
    covariance_data = np.einsum('ij,ki->jk', X_centered, X_centered.T) / (n_samples - 1)
    bandwidth = 1 / (100 * n_features) * np.trace(covariance_data)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    return kde.score_samples(X)


def _find_density_diagonal_bandwidth(X, n_features):
    bandwidths = np.array([0.01 * np.std(X[:, i]) for i in range(n_features)])
    var_type = 'c' * n_features
    dens_u = sm.nonparametric.KDEMultivariate(data=X, var_type=var_type, bw=bandwidths)
    return dens_u.pdf(X)


def _find_density_normal_reference_bandwidth(X, n_features):
    var_type = 'c' * n_features
    dens_u = sm.nonparametric.KDEMultivariate(data=X, var_type=var_type, bw='normal_reference')
    return dens_u.pdf(X)


def _find_density_int_bandwidth(X, bandwidth):
    kdt = KDTree(X, metric='euclidean')
    distances, neighbors = kdt.query(X, int(bandwidth))
    return 1 / distances[:, int(bandwidth) - 1]


def _find_density_float_bandwidth(X, bandwidth):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    return kde.score_samples(X)


def _find_density(X, bandwidth, n_samples, n_features):
    if bandwidth == "spherical":
        return _find_density_spherical_bandwidth(X, n_samples, n_features)
    elif bandwidth == "diagonal":
        return _find_density_diagonal_bandwidth(X, n_features)
    elif bandwidth == "normal_reference":
        return _find_density_normal_reference_bandwidth(X, n_features)
    elif isinstance(bandwidth, int):
        return _find_density_int_bandwidth(X, bandwidth)
    elif isinstance(bandwidth, float):
        return _find_density_float_bandwidth(X, bandwidth)


def _initialise_distance(X, n_samples, density):
    kdt = KDTree(X, metric='euclidean')
    distances, neighbors = kdt.query(X, np.floor(np.sqrt(n_samples)).astype(int))
    best_distance = np.empty((X.shape[0]))
    radius_diff = density[:, np.newaxis] - density[neighbors]
    rows, cols = np.where(radius_diff < 0)
    rows, unidx = np.unique(rows, return_index=True)
    cols = cols[unidx]
    best_distance[rows] = distances[rows, cols]
    return best_distance, rows


def _find_gt_radius(density, search_idx):
    search_density = density[search_idx]
    return density > search_density[:, np.newaxis]


def _radius_sum_is_zero(X, best_distance, search_idx, GT_radius):
    if any(np.sum(GT_radius, axis=1) == 0):
        max_i = [i for i in range(GT_radius.shape[0]) if np.sum(GT_radius[i, :]) == 0]
        if len(max_i) > 1:
            for max_j in max_i[1:len(max_i)]:
                GT_radius[max_j, search_idx[max_i[0]]] = True
        max_i = max_i[0]
        best_distance[search_idx[max_i]] = np.sqrt(((X - X[search_idx[max_i], :]) ** 2).sum(1)).max()
        GT_radius = np.delete(GT_radius, max_i, 0)
        del search_idx[max_i]
    return best_distance, search_idx, GT_radius


def _calculate_distance(X, search_idx, best_distance, GT_radius):
    GT_distances = ([X[search_idx[i], np.newaxis], X[GT_radius[i, :], :]] for i in range(len(search_idx)))
    distances_bb = list(map(_density_broad_search_star, list(GT_distances)))
    argmin_distance = [np.argmin(l) for l in distances_bb]
    for i in range(GT_radius.shape[0]):
        best_distance[search_idx[i]] = distances_bb[i][argmin_distance[i]]
    return best_distance


def _find_distance(X, n_samples, density):
    best_distance, rows = _initialise_distance(X, n_samples, density)
    search_idx = list(np.setdiff1d(list(range(X.shape[0])), rows))
    GT_radius = _find_gt_radius(density, search_idx)
    best_distance, search_idx, GT_radius = _radius_sum_is_zero(X, best_distance, search_idx, GT_radius)
    return _calculate_distance(X, search_idx, best_distance, GT_radius)


def _estimate_density_distances(X, bandwidth):
    n_samples, n_features = X.shape
    density = _find_density(X, bandwidth, n_samples, n_features)
    best_distance = _find_distance(X, n_samples, density)
    return density, best_distance

def _initialize_covariances(X, means, covariance_type):
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    center = X.sum(0) / n_samples

    X_centered = X - center

    covariance_data = np.einsum('ij,ki->jk', X_centered, X_centered.T) / (n_samples - 1)

    variance = 1 / (n_components * n_features) * np.trace(covariance_data)

    if covariance_type == "full":
        covariances = np.stack([np.diag(np.ones(n_features) * variance) for _ in range(n_components)])
    elif covariance_type == "spherical":
        covariances = np.repeat(variance, n_components)
    elif covariance_type == "tied":
        covariances = np.diag(np.ones(n_features) * variance)
    elif covariance_type == "diag":
        covariances = np.ones((n_components, n_features)) * variance
    return covariances

def _get_intervals(n_samples, ltidx, gtidx, ranges):
    raw_intervals = []
    union_intervals = []
    for s in range(n_samples):
        raw_intervals.append([])
        union_intervals.append([])
        for t in ltidx:
            raw_intervals[s].append((-np.inf, ranges[s, t]))
        for t in gtidx:
            raw_intervals[s].append((ranges[s, t], np.inf))
        for begin, end in sorted(raw_intervals[s]):
            if union_intervals[s] and union_intervals[s][-1][1] >= begin - 1:
                union_intervals[s][-1][1] = max(union_intervals[s][-1][1], end)
            else:
                union_intervals[s].append([begin, end])
    return [item for sublist in union_intervals for item in sublist]


def _select_exemplars_from_thresholds(X, density, distance, density_threshold, distance_threshold):
    density_inlier = density > density_threshold
    distance_inlier = distance > distance_threshold
    means_idx = np.where(density_inlier * distance_inlier)[0]
    remainder_idx = np.where(~(density_inlier * distance_inlier))[0]
    means = X[means_idx, :]
    X_iter = X[remainder_idx, :]
#     print("%s modes selected." % means.shape[0])
    return X_iter, means


def _select_exemplars_fromK(X, density, distance, max_components):
    n_samples, _ = X.shape
    means_idx = np.argsort(- density * distance)[range(max_components)]
    remainder_idx = np.argsort(- density * distance)[range(max_components, n_samples)]
    means = X[means_idx, :]
    X_iter = X[remainder_idx, :]
#     print("%s modes selected." % means.shape[0])
    return X_iter, means

def _expand_covariance_matrix(covariances, covariance_type, n_features, n_components):
    if covariance_type == 'spherical':
        return np.array([np.diag(np.ones(n_features) * i) for i in covariances])
    elif covariance_type == 'diag':
        return np.array([np.diag(i) for i in covariances])
    elif covariance_type == 'tied':
        return np.array([covariances] * n_components)
    else:
        return covariances


class overlap:
    def __init__(
        self,
        *,
        data,
        covariance_type="full",
        criteria="all",
        bandwidth="diagonal",
        max_components=5,
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
    ):
        self.data = data
        self.fitted = False
        self.covariance_type = covariance_type
        self.criteria = criteria
        self.bandwidth = bandwidth
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self._distance = None
        self._density = None
        self._density_threshold = None
        self._max_components = max_components
        self._distance_threshold = None
        
    def _initialize_parameters(self):
        self.X_iter, self.means_iter = self._get_exemplars()
        self.n_components_iter = self.means_iter.shape[0]
        self.covariances_iter = _initialize_covariances(self.data, self.means_iter, self.covariance_type)
        self.weights_iter = np.ones((self.n_components_iter)) / self.n_components_iter
#         self._add_mixture()
        
        
    def _get_exemplars(self):
        if self._density is None and self._distance is None:
            self._density, self._distance = _estimate_density_distances(self.data, self.bandwidth)
        return self._select_exemplars()

    
    def _compute_overlap(self, n_features, cov):
        covariances_jitter = self._update_covariance_for_overlap(n_features, cov)
        return self._get_omega_map(n_features, covariances_jitter)
    
    def _update_covariance_for_overlap(self, n_features, cov):
        covariances_jitter = np.zeros(cov.shape)
        for i in range(self.n_components_iter):
            val, vec = np.linalg.eig(cov[i])
            val += np.abs(np.random.normal(loc=0, scale=0.01, size=n_features))
            covariances_jitter[i, :, :] = vec.dot(np.diag(val)).dot(np.linalg.inv(vec))
        return covariances_jitter
    
    
    def _get_omega_map(self, n_features, covariances_jitter):
        while True:
            n_components, _, _ = covariances_jitter.shape
            omega_map = Overlap(n_features, n_components, self.weights_iter, self.means_iter, 
                                covariances_jitter,
                                np.array([1e-06, 1e-06]), 1e06).omega_map
            if np.max(omega_map.max(1)) > 0:
                break
            else:
                covariances_jitter *= 1.1
        self.omega_map = omega_map
        return omega_map.max(1)
    
    
    def _select_exemplars(self):
        if self._density_threshold is not None and self._distance_threshold is not None:
            return _select_exemplars_from_thresholds(self.data, self._density, self._distance, self._density_threshold,
                                                     self._distance_threshold)
        else:
            if self._density_threshold is not None or self._distance_threshold is not None:
                self._print_threshold_parameter_warning()
                
#             print(self._distance)
#             print(self._max_components)
            return _select_exemplars_fromK(self.data, self._density, self._distance, 
                                           self._max_components)
    
    def _get_distances(self, n_samples, covariances):
        distances = np.zeros((n_samples, self.n_components_iter))
        for j in range(self.n_components_iter):
            distances[:, j, np.newaxis] = distance.cdist(self.X_iter, self.means_iter[j, :][np.newaxis],
                                                         metric='mahalanobis', VI=covariances[j, :, :])
        return distances
    
    
    def _get_theta(self, distances, covariances_logdet_penalty, overlap_max):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._compute_theta(distances, covariances_logdet_penalty, overlap_max)
        
    def _alter_covariances(self, n_features, n_samples):
        self.covariances_iter += np.ones(self.covariances_iter.shape) * 1e-6
        expanded_covariance_iter = _expand_covariance_matrix(self.covariances_iter, self.covariance_type, n_features,
                                                             self.n_components_iter)
        covariances_logdet_penalty = np.array(
            [np.log(np.linalg.det(expanded_covariance_iter[i])) for i in range(self.n_components_iter)]) / n_samples
        return covariances_logdet_penalty, expanded_covariance_iter
        
        
    def _compute_theta(self, distances, covariances_logdet_penalty, overlap_max):
        n_samples, _ = self.X_iter.shape
        thetas = np.ones(self.n_components_iter) * np.nan
        entry = False
        for i in range(self.n_components_iter):
            p = distances + covariances_logdet_penalty - (distances[:, i] + covariances_logdet_penalty[i])[:,
                                                         np.newaxis]
            ranges = p / (overlap_max[i] - overlap_max)
            noni_idx = list(range(self.n_components_iter))
            noni_idx.pop(i)
            overlap_noni = overlap_max[noni_idx]
            ranges = ranges[:, noni_idx]
            ltidx = np.where(overlap_max[i] < overlap_noni)[0]
            gtidx = np.where(overlap_max[i] > overlap_noni)[0]
            union_intervals = _get_intervals(n_samples, ltidx, gtidx, ranges)
            start, end = None, None
            while union_intervals:
                start_temp, end_temp = union_intervals.pop()
                start = start_temp if start is None else max(start, start_temp)
                end = end_temp if end is None else min(end, end_temp)
            if start is not None and end is not None and end > start > 0:
                entry = True
                thetas[i] = start
        if not entry:
            return None
        theta = thetas[~np.isnan(thetas)].min()
        return theta * 1.0001
    
    def _return_refined(self, resps):
        self._update_weights(resps)
        self._add_exemplar_to_data()
        self._remove_pruned_mean()
        self._remove_pruned_covariance()
        self._remove_pruned_component()
        
    def _get_pruning_parameters(self):
        n_samples, n_features = self.X_iter.shape
        covariances_logdet_penalty, expanded_covariance_iter = self._alter_covariances(n_features, n_samples)
        overlap_max = self._compute_overlap(n_features, expanded_covariance_iter)
        distances = self._get_distances(n_samples, expanded_covariance_iter)
        theta = self._get_theta(distances, covariances_logdet_penalty, overlap_max)
        return n_samples, covariances_logdet_penalty, overlap_max, distances, theta
    
    def get_overlap(self):
        self._initialize_parameters()
        n_samples, n_features = self.X_iter.shape
        covariances_logdet_penalty, expanded_covariance_iter = self._alter_covariances(n_features, n_samples)
        overlap_max = self._compute_overlap(n_features, expanded_covariance_iter)
        distances = self._get_distances(n_samples, expanded_covariance_iter)
        theta = self._get_theta(distances, covariances_logdet_penalty, overlap_max)
        return overlap_max
    
        
    def get_result(self):
        self._initialize_parameters()
        return self._get_pruning_parameters()


class unigmm:
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
    
    def get_overlap(self):
        tem = overlap(data = self.projection_coefficients, max_components = self.n_components)
        self.overlap_max = tem.get_overlap()
        self.omega_map = tem.omega_map
        return self.omega_map
