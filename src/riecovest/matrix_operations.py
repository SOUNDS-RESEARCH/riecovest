"""Functions for matrix operations using jax.

Made to ease the use of jax in algorithms requiring automatic differentiation, by adding some common operations that does not already exist in the library. 
"""


import jax.numpy as jnp


def matrix_sqrt(A):
    """Computes the square root of a positive definite matrix A.

    Clips eigenvalues below 1e-12 to avoid numerical instability.

    Parameters
    ----------
    A : ndarray of shape (M, M)
        Positive definite matrix
    
    Returns
    -------
    sqrt_A : ndarray of shape (M, M)
        The square root of A. Such that sqrt_A @ sqrt_A = A
    
    """
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-12)
    return eigvecs @ jnp.sqrt(jnp.diag(eigvals)) @ jnp.conj(eigvecs).T


def generalized_eigh(A, B):
    """Computes the generalized eigenvalue decomposition of a pair of positive semidefinite matrices.
    
    Returns the same as scipy.linalg.eigh(A, B)

    Parameters
    ----------
    A : ndarray of shape (M, M)
        Hermitian matrix
    B : ndarray of shape (M, M)
        Positive definite matrix

    Returns
    -------
    eigenvalues : ndarray of shape (M,)
        Eigenvalues in ascending order
    eigenvectors : ndarray of shape (M, M)
        Eigenvectors in the columns
    """
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ jnp.conj(L_inv.T)
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)
    eigenvectors_original = jnp.conj(L_inv.T) @ eigenvectors_transformed
    return eigenvalues, eigenvectors_original

