import jax.numpy as jnp
import numpy as np
import scipy.linalg as splin

import pymanopt


# ========================== DISTANCE METRICS ==========================

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
    A_sqrt = matrix_sqrt(A)
    mix_term = A_sqrt @ B @ A_sqrt
    mix_term_sqrt = matrix_sqrt(mix_term)
    return jnp.real(jnp.trace(A + B - 2 * mix_term_sqrt))

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
    eigvals, _ = generalized_eigh(A, B)
    return jnp.real(jnp.sqrt(jnp.sum(jnp.log(eigvals)**2)))

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

def frob_gevd_weighted(A, B): 
    eigvals, eigvec = generalized_eigh(A, B)
    eigvec = jnp.flip(eigvec, axis=-1)
    eigvals = jnp.flip(eigvals, axis=-1)
    W = eigvec.T.conj()

    diff = W @ (A - B) @ W.T.conj()
    return jnp.real(jnp.sqrt(jnp.trace(diff @ diff.conj().T)))



def wishart_likelihood(mat_variable, cov, N):
    """Wishart likelihood function.
    
    It is not a true likelihood, since the mass is not 1. It is however proportional to true likelihood with regards to the covariance matrix. Anything constant with regards
    to the covariance is not taken into account. Therefore, the maximum likelihood estimator can be found by maximizing this function.

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

    It is not a true likelihood, since the mass is not 1. It is however proportional to true likelihood with regards to the covariance matrix. Anything constant with regards
    to the covariance is not taken into account. Therefore, the maximum likelihood estimator can be found by maximizing this function.

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


















#====================================== COVARIANCE ESTIMATORS ======================================
def scm(data):
    """Compute the sample covariance matrix of the given data.

    Parameters
    ----------
    data : ndarray of shape (dim, num_samples)
        Data matrix.
    
    Returns
    -------
    scm : ndarray of shape (dim, dim)
        Sample covariance matrix.
    """
    return data @ data.conj().T / data.shape[1]


def tyler_estimator(data, num_iter=20):
    """Compute the Tyler estimator of the covariance matrix of the given data.

    Parameters
    ----------
    data : ndarray of shape (dim, num_samples)
        Data matrix.
    num_iter : int
        Number of iterations of the Tyler estimator algorithm.

    Returns
    -------
    tyler : ndarray of shape (dim, dim)
        Tyler estimator of the covariance matrix.
    """
    dim = data.shape[0]
    num_samples = data.shape[1]
    normalized_data = data / np.linalg.norm(data, axis=0, keepdims=True)

    cov = np.eye(dim, dtype=complex)
    for _ in range(num_iter):
        cov_inv = np.linalg.inv(cov)
        #tyler = jnp.zeros_like(tyler)

        update = np.zeros_like(cov, dtype=complex)
        for i in range(num_samples):
            x = data[:,i:i+1]
            update += x @ x.T.conj() / (x.T.conj() @ cov_inv @ x)

        update *= dim / np.trace(update)
        cov = update
    return cov



#==================== NOISE PLUS SIGNAL COVARIANCE ESTIMATORS

# def tyler_estimator_noisy_signal(data_noisy_signal, data_noise, num_iter=20):
#     Ry = tyler_estimator(data_noisy_signal, num_iter=num_iter)
#     Rv = tyler_estimator(data_noise, num_iter=num_iter)
#     return Ry - Rv, Rv



def est_gevd(Ry, Rn, rank, est_noise_cov=False):
    if Ry.ndim == 2:
        return _est_gevd(Ry, Rn, rank, est_noise_cov=est_noise_cov)
    
    out = np.zeros_like(Ry)
    if est_noise_cov:
        out_noise_cov = np.zeros_like(Ry)
        for f in range(Ry.shape[0]):
            _est_gevd(Ry[f,:,:], Rn[f,:,:], rank, out=out[f,:,:], est_noise_cov=est_noise_cov, out_noise_cov=out_noise_cov)
        if np.isnan(np.sum(out)):
            print(f"#warning")
        return out, out_noise_cov
    
    for f in range(Ry.shape[0]):
        _est_gevd(Ry[f,:,:], Rn[f,:,:], rank, out=out[f,:,:])
    return out

def _est_gevd(Ry, Rn, rank, out=None, est_noise_cov=False, out_noise_cov=None):
    dim = Rn.shape[0]

    eigvals, eigvec = splin.eigh(Ry, Rn + 1e-10*np.eye(dim))
    eigvec = np.flip(eigvec, axis=-1)
    eigvec_invt = splin.inv(eigvec).T.conj()
    eigvals = np.flip(eigvals, axis=-1)

    new_eigvals = eigvals[:rank]
    new_eigvals = new_eigvals - np.ones_like(new_eigvals)
    np.maximum(new_eigvals, np.zeros_like(new_eigvals), out=new_eigvals)
    adjusted_rank = np.count_nonzero(new_eigvals)

    if out is None:
        out = np.zeros_like(Ry)
    else:
        out.fill(0)

    for r in range(adjusted_rank):
        out += new_eigvals[r] * eigvec_invt[:,r:r+1] @ eigvec_invt[:,r:r+1].T.conj()

    if est_noise_cov:
        if out_noise_cov is None:
            out_noise_cov = np.zeros_like(Ry)
        else:
            out_noise_cov.fill(0)
        noise_eigvals = np.zeros_like(eigvals)
        noise_eigvals[:adjusted_rank] = 1
        noise_eigvals[adjusted_rank:] = (eigvals[adjusted_rank:] + 1) / 2 #here we set alpha=0.5, see paper by Serizel

        for r in range(dim):
            out_noise_cov += noise_eigvals[r] * eigvec_invt[:,r:r+1] @ eigvec_invt[:,r:r+1].T.conj()
        return out, out_noise_cov
    else:
        return out


def est_covariance_manifold(ambient_dim, rank, cost, manifold, start_value=None):
    problem = pymanopt.Problem(
        manifold,
        cost,
    )
    optimizer = pymanopt.optimizers.TrustRegions(
        max_iterations=500, verbosity=0
    )
    if start_value is None:
        cov_est = optimizer.run(problem, Delta_bar=8*np.sqrt(rank)).point
    else:
        u, s, vh = np.linalg.svd(start_value[0], full_matrices=True, hermitian=True)
        init_rx = u[:,:rank] * np.sqrt(s[None,:rank])
        start_value = (init_rx, start_value[1])
        cov_est = optimizer.run(problem, Delta_bar=8*np.sqrt(rank), initial_point=start_value).point
    X, Rv = cov_est
    Rx = X @ jnp.conj(X.T)
    return Rx, Rv

def est_manifold_frob(scm_noisy_signal, scm_noise, rank, start_value=None):
    ambient_dim = scm_noisy_signal.shape[0]
    scm_noisy_signal = (scm_noisy_signal + scm_noisy_signal.conj().T) / 2
    scm_noise = (scm_noise + scm_noise.conj().T) / 2
    
    signal_manifold = pymanopt.manifolds.PSDFixedRankComplex(ambient_dim, rank)
    noise_manifold = pymanopt.manifolds.HermitianPositiveDefinite(ambient_dim)
    joint_manifold = pymanopt.manifolds.Product([signal_manifold, noise_manifold])

    @pymanopt.function.jax(joint_manifold)
    def cost(X, Rv):
        """
        X is shape : (ambient_dim, rank)
        V is shape : (ambient_dim, ambient_dim)
        """
        Rx = X @ jnp.conj(X.T)
        Ry = Rx + Rv
        #error_y = scm_noisy_signal - Ry
        #error_v = scm_noise - Rv
        return frob_sq(scm_noisy_signal, Ry) + frob_sq(scm_noise, Rv)
    
    return est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)

def est_manifold_wishart(scm_noisy_signal, scm_noise, rank, num_scm_samples, alpha=0.5, start_value=None):
    ambient_dim = scm_noisy_signal.shape[0]
    scm_noisy_signal = (scm_noisy_signal + scm_noisy_signal.conj().T) / 2
    scm_noise = (scm_noise + scm_noise.conj().T) / 2
    
    signal_manifold = pymanopt.manifolds.PSDFixedRankComplex(ambient_dim, rank)
    noise_manifold = pymanopt.manifolds.HermitianPositiveDefinite(ambient_dim)
    joint_manifold = pymanopt.manifolds.Product([signal_manifold, noise_manifold])

    @pymanopt.function.jax(joint_manifold)
    def cost(X, Rv):
        """
        X is shape : (ambient_dim, rank)
        V is shape : (ambient_dim, ambient_dim)
        """
        Rx = X @ jnp.conj(X.T)
        Ry = Rx + Rv
        
        llh_noisy_sig = wishart_log_likelihood(scm_noisy_signal, Ry, num_scm_samples)
        llh_noise = wishart_log_likelihood(scm_noise, Rv, num_scm_samples)
        #print(f"{llh_noisy_sig}, {llh_noise}")
        return -alpha * llh_noisy_sig - (1-alpha) * llh_noise
    Rx_est, Rv_est = est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)
    #scaling = np.trace(scm_noisy_signal) / np.trace(Rx_est + Rv_est)
    #Rx_est *= scaling
    #Rv_est *= scaling
    return Rx_est, Rv_est



def est_manifold_tylers_m_estimator(data_noisy_signal, data_noise, rank, start_value=None):
    ambient_dim = data_noisy_signal.shape[0]

    scaling = np.trace(scm(data_noisy_signal))
    # scm_noisy_signal = (scm_noisy_signal + scm_noisy_signal.conj().T) / 2
    # scm_noise = (scm_noise + scm_noise.conj().T) / 2
    
    signal_manifold = pymanopt.manifolds.PSDFixedRankComplex(ambient_dim, rank)
    noise_manifold = pymanopt.manifolds.HermitianPositiveDefinite(ambient_dim)
    joint_manifold = pymanopt.manifolds.Product([signal_manifold, noise_manifold])

    @pymanopt.function.jax(joint_manifold)
    def cost(X, Rv):
        """
        X is shape : (ambient_dim, rank)
        V is shape : (ambient_dim, ambient_dim)
        """
        Rx = X @ jnp.conj(X.T)
        Ry = Rx + Rv
        
        #scaling_factor = scaling / jnp.trace(Ry)

        #Rv *= scaling_factor
        #Rx *= scaling_factor
        #Ry *= scaling_factor
        
        llh_noisy_sig = tyler_log_likelihood(data_noisy_signal, Ry)
        llh_noise = tyler_log_likelihood(data_noise, Rv)
        return -llh_noisy_sig - llh_noise 
    Rx_est, Rv_est = est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)

    #scaling_factor = ambient_dim / np.trace(Rx_est)
    scaling = np.trace(scm(data_noisy_signal)) / np.trace(Rx_est + Rv_est)
    Rx_est *= scaling
    Rv_est *= scaling
    return Rx_est, Rv_est


def est_manifold_tylers_m_estimator_nonorm(data_noisy_signal, data_noise, rank, start_value=None):
    ambient_dim = data_noisy_signal.shape[0]

    trace_noisy_sig = np.trace(scm(data_noisy_signal))
    trace_noise = np.trace(scm(data_noise))
    trace_signal = trace_noisy_sig - trace_noise
    # scm_noisy_signal = (scm_noisy_signal + scm_noisy_signal.conj().T) / 2
    # scm_noise = (scm_noise + scm_noise.conj().T) / 2
    
    signal_manifold = pymanopt.manifolds.PSDFixedRankComplex(ambient_dim, rank)
    noise_manifold = pymanopt.manifolds.HermitianPositiveDefinite(ambient_dim)
    joint_manifold = pymanopt.manifolds.Product([signal_manifold, noise_manifold])

    @pymanopt.function.jax(joint_manifold)
    def cost(X, Rv):
        """
        X is shape : (ambient_dim, rank)
        V is shape : (ambient_dim, ambient_dim)
        """
        Rx = X @ jnp.conj(X.T)
        Rx = Rx / jnp.trace(Rx) * trace_signal
        Rv = Rv / jnp.trace(Rv) * trace_noise
        Ry = Rx + Rv
        
        #scaling_factor = scaling / jnp.trace(Ry)

        #Rv *= scaling_factor
        #Rx *= scaling_factor
        #Ry *= scaling_factor
        
        llh_noisy_sig = tyler_log_likelihood_no_normalization(data_noisy_signal, Ry)
        llh_noise = tyler_log_likelihood_no_normalization(data_noise, Rv)
        return -llh_noisy_sig - llh_noise 
    Rx_est, Rv_est = est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)

    #scaling_factor = ambient_dim / np.trace(Rx_est)
    #scaling = np.trace(scm(data_noisy_signal)) / np.trace(Rx_est + Rv_est)
    Rx_est = Rx_est / np.trace(Rx_est) * trace_signal
    Rv_est = Rv_est / np.trace(Rv_est) * trace_noise
    return Rx_est, Rv_est



def est_manifold_frob_whitened(scm_noisy_signal, scm_noise, rank, alpha = 0.5, start_value=None):
    ambient_dim = scm_noisy_signal.shape[0]
    scm_noisy_signal = (scm_noisy_signal + scm_noisy_signal.conj().T) / 2
    scm_noise = (scm_noise + scm_noise.conj().T) / 2

    # eigvec_invt is Q in the derivations, eigvec is Q^-H
    eigvals, eigvec = splin.eigh(scm_noisy_signal, scm_noise + 1e-10*np.eye(scm_noise.shape[0]))
    #eigvals2, eigvec2 = generalized_eigh(scm_noisy_signal, scm_noise + 1e-10*np.eye(scm_noise.shape[0]))
    eigvec = np.flip(eigvec, axis=-1)
    #eigvec_invt = splin.inv(eigvec).T.conj()
    eigvals = np.flip(eigvals, axis=-1)
    Q_inv = eigvec.T.conj()
    By = np.diag(1 / np.sqrt(eigvals)) @ Q_inv
    Bv = Q_inv

    signal_manifold = pymanopt.manifolds.PSDFixedRankComplex(ambient_dim, rank)
    noise_manifold = pymanopt.manifolds.HermitianPositiveDefinite(ambient_dim)
    joint_manifold = pymanopt.manifolds.Product([signal_manifold, noise_manifold])

    @pymanopt.function.jax(joint_manifold)
    def cost(X, Rv):
        """
        X is shape : (ambient_dim, rank)
        V is shape : (ambient_dim, ambient_dim)
        """
        Rx = X @ jnp.conj(X.T)
        Ry = Rx + Rv
        #error_y = scm_noisy_signal - Ry
        #error_v = scm_noise - Rv
        return alpha * frob_sq_weighted(scm_noisy_signal, Ry, Bv) + (1-alpha) * frob_sq_weighted(scm_noise, Rv, Bv)
    return est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)
    # problem = pymanopt.Problem(
    #     joint_manifold,
    #     cost,
    # )

    # optimizer = pymanopt.optimizers.TrustRegions(
    #     max_iterations=100, verbosity=2
    # )
    # if start_value is None:
    #     cov_est = optimizer.run(problem, Delta_bar=8*np.sqrt(rank)).point
    # else:
    #     u, s, vh = np.linalg.svd(start_value[0], full_matrices=True, hermitian=True)
    #     init_rx = u[:,:rank] * np.sqrt(s[None,:rank])
    #     start_value = (init_rx, start_value[1])
    #     cov_est = optimizer.run(problem, Delta_bar=8*np.sqrt(rank), initial_point=start_value).point
    # X, Rv = cov_est
    # Rx = X @ jnp.conj(X.T)
    # return Rx, Rv


def est_manifold_frob_whitened_by_true_noise(scm_noisy_signal, scm_noise, rank, alpha=0.5, start_value=None):
    """
    Same as est_manifold_frob_whitened, except that we extract the whitening parameter from 
    the optimization variable Rv instead of the sample covariance matrix. It should according to
    Robbes paper leda to the same solution for frobenious norm distance. 
    """
    ambient_dim = scm_noisy_signal.shape[0]
    scm_noisy_signal = (scm_noisy_signal + scm_noisy_signal.conj().T) / 2
    scm_noise = (scm_noise + scm_noise.conj().T) / 2

    # eigvec_invt is Q in the derivations, eigvec is Q^-H
    #eigvals, eigvec = splin.eigh(scm_noisy_signal, scm_noise + 1e-10*np.eye(scm_noise.shape[0]))
    

    signal_manifold = pymanopt.manifolds.PSDFixedRankComplex(ambient_dim, rank)
    noise_manifold = pymanopt.manifolds.HermitianPositiveDefinite(ambient_dim)
    joint_manifold = pymanopt.manifolds.Product([signal_manifold, noise_manifold])

    @pymanopt.function.jax(joint_manifold)
    def cost(X, Rv):
        """
        X is shape : (ambient_dim, rank)
        V is shape : (ambient_dim, ambient_dim)
        """
        Rx = X @ jnp.conj(X.T)
        Ry = Rx + Rv

        eigvals, eigvec = generalized_eigh(Ry, Rv + 1e-10*np.eye(scm_noise.shape[0]))
        eigvec = jnp.flip(eigvec, axis=-1)
        eigvals = jnp.flip(eigvals, axis=-1)
        Bv = eigvec.T.conj()

        return alpha * frob_sq_weighted(scm_noisy_signal, Ry, Bv) + (1-alpha)*frob_sq_weighted(scm_noise, Rv, Bv)
    return est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)
    # problem = pymanopt.Problem(
    #     joint_manifold,
    #     cost,
    # )

    # optimizer = pymanopt.optimizers.TrustRegions(
    #     max_iterations=100, verbosity=2
    # )
    # if start_value is None:
    #     cov_est = optimizer.run(problem, Delta_bar=8*np.sqrt(rank)).point
    # else:
    #     u, s, vh = np.linalg.svd(start_value[0], full_matrices=True, hermitian=True)
    #     init_rx = u[:,:rank] * np.sqrt(s[None,:rank])
    #     start_value = (init_rx, start_value[1])
    #     cov_est = optimizer.run(problem, Delta_bar=8*np.sqrt(rank), initial_point=start_value).point
    # X, Rv = cov_est
    # Rx = X @ jnp.conj(X.T)
    # return Rx, Rv



def est_manifold_airm(scm_noisy_signal, scm_noise, rank, alpha = 0.5, start_value=None):
    ambient_dim = scm_noisy_signal.shape[0]
    #scm_noisy_signal = (scm_noisy_signal + scm_noisy_signal.conj().T) / 2
    #scm_noise = (scm_noise + scm_noise.conj().T) / 2

    signal_manifold = pymanopt.manifolds.PSDFixedRankComplex(ambient_dim, rank)
    noise_manifold = pymanopt.manifolds.HermitianPositiveDefinite(ambient_dim)
    joint_manifold = pymanopt.manifolds.Product([signal_manifold, noise_manifold])

    @pymanopt.function.jax(joint_manifold)
    def cost(X, Rv):
        """
        X is shape : (ambient_dim, rank)
        V is shape : (ambient_dim, ambient_dim)
        """
        Rx = X @ jnp.conj(X.T)
        Ry = Rx + Rv
        return alpha * airm(Ry, scm_noisy_signal)**2 + (1 - alpha) * airm(Rv, scm_noise)**2 #+ 1e-4*jnp.real(jnp.trace(Rx @ Rx.conj().T)) + 1e-4*jnp.real(jnp.trace(Rv @ Rv.conj().T)) #1e-5 * frob(Rv, scm_noise) + 1e-5 * frob(Ry, scm_noisy_signal)  #+ 1e-6*jnp.real(jnp.trace(Rx @ Rx.conj().T))#+ 1e-6 * frob(Ry, scm_noisy_signal) + 1e-6 * frob(Rv, scm_noise) #jnp.real(jnp.trace(Rx @ Rx.conj().T))
    return est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)
    # problem = pymanopt.Problem(
    #     joint_manifold,
    #     cost,
    # )

    # optimizer = pymanopt.optimizers.TrustRegions(
    #     max_iterations=300, min_step_size=1e-8, verbosity=2
    # )
    # if start_value is None:
    #     cov_est = optimizer.run(problem, Delta_bar=8*np.sqrt(rank)).point
    # else:
    #     u, s, vh = np.linalg.svd(start_value[0], full_matrices=True, hermitian=True)
    #     init_rx = u[:,:rank] * np.sqrt(s[None,:rank])
    #     start_value = (init_rx, start_value[1])
    #     cov_est = optimizer.run(problem, Delta_bar=8*np.sqrt(rank), initial_point=start_value).point
    # X, Rv = cov_est
    # Rx = X @ jnp.conj(X.T)
    # return Rx, Rv









def est_manifold_tylers_with_scale_opt(data_noisy_signal, data_noise, rank):
    ambient_dim = data_noisy_signal.shape[0]

    Ry_trace = np.trace(scm(data_noisy_signal))
    Rv_trace = np.trace(scm(data_noise))
    # scm_noisy_signal = (scm_noisy_signal + scm_noisy_signal.conj().T) / 2
    # scm_noise = (scm_noise + scm_noise.conj().T) / 2
    
    signal_manifold = pymanopt.manifolds.PSDFixedRankComplex(ambient_dim, rank)
    noise_manifold = pymanopt.manifolds.HermitianPositiveDefinite(ambient_dim)
    scaling_manifold = pymanopt.manifolds.Positive(2, 1)
    joint_manifold = pymanopt.manifolds.Product([signal_manifold, noise_manifold, scaling_manifold])

    @pymanopt.function.jax(joint_manifold)
    def cost(X, Rv, scale):
        """
        X is shape : (ambient_dim, rank)
        V is shape : (ambient_dim, ambient_dim)
        """
        Rx = X @ jnp.conj(X.T)
        Rx = Rx * scale[0] / jnp.trace(Rx)
        Rv = Rv * scale[1] / jnp.trace(Rv)
        Ry = Rx + Rv
        
        scaling_error = jnp.abs(jnp.trace(Rx + Rv) - Ry_trace)**2 + jnp.abs(jnp.trace(Rv) - Rv_trace)**2

        #scaling_factor = scaling / jnp.trace(Ry)

        #Rv *= scaling_factor
        #Rx *= scaling_factor
        #Ry *= scaling_factor
        
        llh_noisy_sig = tyler_log_likelihood(data_noisy_signal, Ry)
        llh_noise = tyler_log_likelihood(data_noise, Rv)
        return -llh_noisy_sig - llh_noise + 1e3 * scaling_error
    
    problem = pymanopt.Problem(joint_manifold,cost,)
    optimizer = pymanopt.optimizers.TrustRegions(max_iterations=100, verbosity=2)
    cov_est = optimizer.run(problem, Delta_bar=8*np.sqrt(rank)).point
    X, Rv, scale = cov_est
    Rx = X @ jnp.conj(X.T)
    Rx *= scale[0] / jnp.trace(Rx)
    Rv *= scale[1] / jnp.trace(Rv)


    #Rx_est, Rv_est = est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)
    #scaling_factor = ambient_dim / np.trace(Rx_est)
    #scaling = np.trace(scm(data_noisy_signal)) / np.trace(Rx_est + Rv_est)
    #Rx_est *= scaling
    #Rv_est *= scaling
    return Rx, Rv



