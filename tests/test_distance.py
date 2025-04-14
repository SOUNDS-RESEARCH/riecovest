import numpy as np
import jax

import scipy.linalg as splin

import riecovest.distance as dist

def get_random_pd_mat(mat_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    rank = 2 * mat_size
    R = np.zeros((mat_size, mat_size), dtype=complex)
    for r in range(rank):
        v = rng.normal(size=(mat_size, 1)) + 1j*rng.normal(size=(mat_size, 1))
        R += v @ v.conj().T

    assert np.allclose(R, R.conj().T)
    return R

def test_gevd_distance_with_full_rank_equals_old_implementation():
    dim = 5
    A = get_random_pd_mat(dim)
    B = get_random_pd_mat(dim)

    with jax.disable_jit():
        distance1 = dist.frob_gevd_weighted_fullrank(A, B)
        distance2 = dist.frob_gevd_weighted(A, B, rank = dim)

    assert np.allclose(distance1, distance2)



def test_airm_equals_numpy_implementation():
    dim = 5
    A = get_random_pd_mat(dim)
    B = get_random_pd_mat(dim)

    #with jax.disable_jit():
    distance1 = dist.airm(A, B)
    distance2 = _numpy_airm(A, B)

    assert np.allclose(distance1, distance2)



def test_corr_matrix_distance_equals_numpy_implementation():
    dim = 5
    A = get_random_pd_mat(dim)
    B = get_random_pd_mat(dim)

    #with jax.disable_jit():
    distance1 = dist.corr_matrix_distance(A, B)
    distance2 = _numpy_corr_matrix_distance(A, B)

    assert np.allclose(distance1, distance2)


def test_kl_divergence_gaussian_equals_numpy_implementation():
    dim = 5
    A = get_random_pd_mat(dim)
    B = get_random_pd_mat(dim)

    #with jax.disable_jit():
    distance1 = dist.kl_divergence_gaussian(A, B)
    distance2 = _numpy_covariance_distance_kl_divergence(A, B)

    assert np.allclose(distance1, distance2)

























def _numpy_corr_matrix_distance(mat1, mat2):
    """Computes the correlation matrix distance
    
    0 means that the matrices are equal up to a scaling
    1 means that they are maximally different (orthogonal in NxN dimensional space)

    Parameters
    ----------
    mat1 : np.ndarray of shape (..., N, N)
        First covariance matrix, should be symmetric and positive definite
    mat2 : np.ndarray of shape (..., N, N)
        Second covariance matrix, should be symmetric and positive definite

    References
    ----------
    Correlation matrix distaince, a meaningful measure for evaluation of 
    non-stationary MIMO channels - Herdin, Czink, Ozcelik, Bonek
    """
    assert mat1.shape == mat2.shape
    norm1 = np.linalg.norm(mat1, ord="fro", axis=(-2,-1))
    norm2 = np.linalg.norm(mat2, ord="fro", axis=(-2,-1))
    if norm1 * norm2 == 0:
        return np.array(np.nan)
    return np.real_if_close(1 - np.trace(mat1 @ mat2) / (norm1 * norm2))


def _numpy_airm(mat1, mat2):
    """
    Computes the covariance matrix distance

    Parameters
    ----------
    mat1 : np.ndarray of shape (N, N)
        First covariance matrix, should be symmetric and positive definite
    mat2 : np.ndarray of shape (N, N)
        Second covariance matrix, should be symmetric and positive definite

    Returns
    -------
    dist : float
        The distance between the two matrices
    
    Notes
    -----
    It is the distance of a canonical invariant Riemannian metric on the space 
    Sym+(n, R) of real symmetric positive definite matrices. 

    Invariant to affine transformations and inversions. 
    It is a distance measure, so 0 means equal and then it goes to infinity
    and the matrices become more unequal.

    When the metric of the space is the fisher information metric, this is the 
    distance of the space. See COVARIANCE CLUSTERING ON RIEMANNIAN MANIFOLDS
    FOR ACOUSTIC MODEL COMPRESSION - Shinohara, Masukp, Akamine

    References
    ----------
    [forstnermetric2003]
    [absilOptimization2008]
    """
    assert mat1.shape == mat2.shape
    assert mat1.shape[0] == mat1.shape[1]
    assert mat1.ndim == 2
    eigvals = splin.eigh(mat1, mat2, eigvals_only=True)
    return np.real_if_close(np.sqrt(np.sum(np.log(eigvals)**2)))


def _numpy_covariance_distance_kl_divergence(mat1, mat2):
    """The Kullback Leibler divergence between two Gaussian
    distributions that has mat1 and mat2 as their covariance matrices. 

    Assumes both of these distributions has zero mean.

    It is a distance measure, so 0 means equal and then it goes to infinity
    and the matrices become more unequal.

    Parameters
    ----------
    mat1 : np.ndarray of shape (N, N)
        First covariance matrix, should be symmetric and positive definite
    mat2 : np.ndarray of shape (N, N)
        Second covariance matrix, should be symmetric and positive definite
    
    Returns
    -------
    dist : float
        The distance between the two matrices
    
    """
    assert mat1.shape == mat2.shape
    assert mat1.shape[0] == mat1.shape[1]
    assert mat1.ndim == 2
    N = mat1.shape[0]
    eigvals = splin.eigh(mat1, mat2, eigvals_only=True)

    det1 = splin.det(mat1)
    det2 = splin.det(mat2)
    common_trace = np.sum(eigvals)
    return np.real_if_close(np.sqrt((np.log(det2 / det1) + common_trace - N) / 2))
