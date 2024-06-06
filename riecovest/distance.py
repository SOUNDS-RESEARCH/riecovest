"""Functions for computing distances between matrices. Most of the functions are distances between positive definite matrices, although some are for general matrices.

All operations are defined in jax, so that they can be used in algorithms requiring automatic differentiation.
"""


import jax.numpy as jnp
import matrix_operations as matop


def frob_sq(A, B):
    """The squared Frobenius distance between two matrices A and B.
    
    Parameters
    ----------
    A : ndarray of shape (M, N)
        Matrix
    B : ndarray of shape (M, N)
        Matrix
    
    Returns
    -------
    distance : float
        The squared Frobenius distance between A and B
    """
    diff = A - B
    return jnp.real(jnp.trace(diff @ diff.conj().T))

def frob_sq_weighted(A, B, W):
    """The squared Frobenius distance between two matrices A and B, weighted by a matrix W.
    
    Parameters
    ----------
    A : ndarray of shape (M, M)
        Matrix
    B : ndarray of shape (M, M)
        Matrix
    W : ndarray of shape (M, M)
        Weighting matrix

    Returns
    -------
    distance : float
        The squared Frobenius distance between A and B, weighted by W
    """
    diff = W @ (A - B) @ W.T.conj()
    return jnp.real(jnp.trace(diff @ diff.conj().T))

def wasserstein_distance(A, B):
    """Computes the Wasserstein distance between two zero-mean Gaussian distributions defined by two positive definite matrices.

    Parameters
    ----------
    A : ndarray of shape (M, M)
        Positive definite matrix    
    B : ndarray of shape (M, M)
        Positive definite matrix

    Returns
    -------
    distance : float
        The distance between A and B in Wasserstein distance
    """
    A_sqrt = matop.matrix_sqrt(A)
    mix_term = A_sqrt @ B @ A_sqrt
    mix_term_sqrt = matop.matrix_sqrt(mix_term)
    return jnp.real(jnp.trace(A + B - 2 * mix_term_sqrt))

def airm(A, B):
    """The affine invariant Riemannian metric distance between two positive definite matrices.
    
    Parameters
    ----------
    A : ndarray of shape (M, M)
        Positive definite matrix
    B : ndarray of shape (M, M)
        Positive definite matrix
    
    Returns
    -------
    distance : float
        The distance between A and B in AIRM distance
    """
    eigvals, _ = matop.generalized_eigh(A, B)
    return jnp.real(jnp.sqrt(jnp.sum(jnp.log(eigvals)**2)))



def frob_gevd_weighted(A, B):
    """The frobenious distance between A and B, weighted by the generalized eigenvectors. 

    Defined by $\lVert W (A - B) W^H \rVert_F$, where $W$ is the matrix of eigenvectors of the generalized eigenvalue decomposition of $A$ and $B$.
    
    Parameters
    ----------
    A : ndarray of shape (M, M)
        Positive semi-definite matrix
    B : ndarray of shape (M, M)
        Positive definite matrix

    Returns
    -------
    distance : float
        The distance between A and B in weighted Frobenius distance
        
    Notes
    -----
    The defintion of the eigenvector matrix is in terms of the simultaneous diagonalization of $A$ and $B$.
    $A = W \Sigma W^H$
    $B = W W^H$
    """
    eigvals, eigvec = matop.generalized_eigh(A, B)
    eigvec = jnp.flip(eigvec, axis=-1)
    eigvals = jnp.flip(eigvals, axis=-1)
    W = eigvec.T.conj()

    diff = W @ (A - B) @ W.T.conj()
    return jnp.real(jnp.sqrt(jnp.trace(diff @ diff.conj().T)))



def wishart_likelihood(mat_variable, cov, N):
    """Wishart likelihood function.
    
    It is not a true likelihood, since the mass is not 1. It is however proportional to true likelihood with regards to the covariance matrix. Anything constant with regards to the covariance is not taken into account. Therefore, the maximum likelihood estimator can be found by maximizing this function.

    Parameters
    ----------
    mat_variable : ndarray of shape (M, M)
        positive definite matrix
    cov : ndarray of shape (M, M)
        positive definite matrix
    N : int
        the degree of freedom parameter for the wishart distribution

    Returns
    -------
    l : float
        The likelihood of the data given the covariance matrix
    """
    M = mat_variable.shape[-1]
    f1 = jnp.linalg.det(cov)**(-N)
    f2 = jnp.exp(-jnp.trace(jnp.linalg.solve(cov, mat_variable)))
    l = f1 * f2
    if (jnp.abs(jnp.imag(l)) / jnp.abs(jnp.real(l))) > 1e-10:
        print(f"warning: more than 1e-10 imaginary part for wishart likelihood")
    return jnp.real(l)

def wishart_log_likelihood(mat_variable, cov, N):
    """Wishart log likelihood function.

    It is not a true likelihood, since the mass is not 1. It is however proportional to true likelihood with regards to the covariance matrix. Anything constant with regards to the covariance is not taken into account. Therefore, the maximum likelihood estimator can be found by maximizing this function.

    Parameters
    ----------
    mat_variable : ndarray of shape (M, M)
        positive definite matrix
    cov : ndarray of shape (M, M)
        positive definite matrix
    N : int
        the degree of freedom parameter for the wishart distribution

    Returns
    -------
    l : float
        The log likelihood of the data given the covariance matrix
    """
    M = mat_variable.shape[-1]
    f1 = -N * jnp.log(jnp.linalg.det(cov))
    f2 = -jnp.trace(jnp.linalg.solve(cov, mat_variable))
    l = f1 + f2
    if (jnp.abs(jnp.imag(l)) / jnp.abs(jnp.real(l))) > 1e-10:
        print(f"warning: more than 1e-10 imaginary part for wishart likelihood")
    return jnp.real(l)



def tyler_log_likelihood(data, cov):
    """Log likelihood function for a complex elliptically symmetric distribution.

    It is not a true likelihood, since the mass is not 1. It is however proportional to true likelihood with regards to the covariance matrix. Anything constant with regards
    to the covariance is not taken into account. Therefore, the maximum likelihood estimator can be found by maximizing this function.

    Parameters
    ----------
    data : ndarray of shape (dim, num_samples)
        data matrix that the likelihood is computed for
    cov : ndrray of shape (dim, dim)
        Covariance matrix of the distibution

    Returns
    -------
    likelihood : float
        The log likelihood of the data given the covariance matrix
        
    References
    ----------
    [ollilaComplex2012] E. Ollila, D. E. Tyler, V. Koivunen, and H. V. Poor, “Complex elliptically symmetric distributions: survey, new results and applications,” IEEE Transactions on Signal Processing, vol. 60, no. 11, pp. 5597–5625, Nov. 2012, doi: 10.1109/TSP.2012.2212433.
    """
    num_samples = data.shape[-1]
    dim = data.shape[0]
    cov_inv = jnp.linalg.inv(cov)
    
    likelihood = 0
    for i in range(num_samples):
        x = data[:,i:i+1]
        x_bar = x / jnp.linalg.norm(x)
        likelihood += jnp.squeeze(jnp.log(jnp.real(x_bar.T.conj() @ cov_inv @ x_bar)) * (dim / num_samples))

    likelihood += jnp.log(jnp.real(jnp.linalg.det(cov)))
    return -likelihood


def tyler_log_likelihood_no_normalization(data, cov):
    """Log likelihood function for a complex elliptically symmetric distribution.

    Same as tyler_log_likelihood, but without normalizing the data to be unit vectors.

    It is not a true likelihood, since the mass is not 1. It is however proportional to true likelihood with regards to the covariance matrix. Anything constant with regards
    to the covariance is not taken into account. Therefore, the maximum likelihood estimator can be found by maximizing this function.

    Parameters
    ----------
    data : ndarray of shape (dim, num_samples)
        data matrix that the likelihood is computed for
    cov : ndrray of shape (dim, dim)
        Covariance matrix of the distibution

    Returns
    -------
    likelihood : float
        The log likelihood of the data given the covariance matrix

    References
    ----------
    [ollilaComplex2012] E. Ollila, D. E. Tyler, V. Koivunen, and H. V. Poor, “Complex elliptically symmetric distributions: survey, new results and applications,” IEEE Transactions on Signal Processing, vol. 60, no. 11, pp. 5597–5625, Nov. 2012, doi: 10.1109/TSP.2012.2212433.
    
    """
    num_samples = data.shape[-1]
    dim = data.shape[0]
    cov_inv = jnp.linalg.inv(cov)
    
    likelihood = 0
    for i in range(num_samples):
        x = data[:,i:i+1]
        likelihood += jnp.squeeze(jnp.log(jnp.real(x.T.conj() @ cov_inv @ x)) * (dim / num_samples))

    likelihood += jnp.log(jnp.real(jnp.linalg.det(cov)))
    return -likelihood

