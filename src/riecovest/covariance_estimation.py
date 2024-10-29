"""Covariance estimation on Riemannian manifolds.

The primary focus is estimating a signal covariance matrix R_x and a noise covariance matrix R_v from a set of noisy signal samples and noise samples. The noise and signal are assumed to be independent, so that the relationship R_y = R_x + R_v holds, where R_y is the covariance of the noisy signal. 

The signal covariance matrix is assumed to be low-rank, while the noise covariance matrix is assumed to be full-rank. 

References
----------
[brunnstroemRobust2024] J. Brunnström, M. Moonen, and F. Elvander, “Robust signal and noise covariance matrix estimation using Riemannian optimization,” presented at the European Signal Processing Conference (EUSIPCO), Sep. 2024 \n
[serizelLowrank2014] R. Serizel, M. Moonen, B. V. Dijk, and J. Wouters, “Low-rank approximation based multichannel Wiener filter algorithms for noise reduction with application in cochlear implants,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 4, pp. 785–799, Apr. 2014, doi: 10.1109/TASLP.2014.2304240.\n
[tylerDistributionfree1987] D. E. Tyler, “A distribution-free M-estimator of multivariate scatter,” The Annals of Statistics, vol. 15, no. 1, pp. 234–251, 1987.
[ollilaComplex2012] E. Ollila, D. E. Tyler, V. Koivunen, and H. V. Poor, “Complex elliptically symmetric distributions: survey, new results and applications,” IEEE Transactions on Signal Processing, vol. 60, no. 11, pp. 5597–5625, Nov. 2012, doi: 10.1109/TSP.2012.2212433. \n
[valrompaeyGEVD2018] R. Van Rompaey and M. Moonen, “GEVD based speech and noise correlation matrix estimation for multichannel Wiener filter based noise reduction,” in 2018 26th European Signal Processing Conference (EUSIPCO), Sep. 2018, pp. 2544–2548. doi: 10.23919/EUSIPCO.2018.8553109. \n
"""

import jax.numpy as jnp
import numpy as np
import scipy.linalg as splin

import pymanopt


import riecovest.distance as dist
import riecovest.matrix_operations as matop

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

    References
    ----------
    [tylerDistributionfree1987] D. E. Tyler, “A distribution-free M-estimator of multivariate scatter,” The Annals of Statistics, vol. 15, no. 1, pp. 234–251, 1987.
    [ollilaComplex2012] E. Ollila, D. E. Tyler, V. Koivunen, and H. V. Poor, “Complex elliptically symmetric distributions: survey, new results and applications,” IEEE Transactions on Signal Processing, vol. 60, no. 11, pp. 5597–5625, Nov. 2012, doi: 10.1109/TSP.2012.2212433.
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
def est_gevd(Ry, Rn, rank, est_noise_cov=False):
    """Signal and noise covariance estimation using the GEVD method.

    Parameters
    ----------
    Ry : ndarray of shape (dim, dim) or (num_freqs, dim, dim)
        Sample covariance matrix of the noisy signal.
    Rn : ndarray of shape (dim, dim) or (num_freqs, dim, dim)
        Sample covariance matrix of the noise.
    rank : int
        Rank of the signal covariance matrix. Must be less than or equal to dim.
    est_noise_cov : bool, optional
        If True, the noise covariance matrix is also estimated. Default is False.

    Returns
    -------
    Rx_hat : ndarray of shape (dim, dim) or (num_freqs, dim, dim)
        Estimated signal covariance matrix.
    Rv_hat : ndarray of shape (dim, dim) or (num_freqs, dim, dim)
        Estimated noise covariance matrix. Only returned if est_noise_cov is True.

    References
    ----------
    [serizelLowrank2014] R. Serizel, M. Moonen, B. V. Dijk, and J. Wouters, “Low-rank approximation based multichannel Wiener filter algorithms for noise reduction with application in cochlear implants,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 4, pp. 785–799, Apr. 2014, doi: 10.1109/TASLP.2014.2304240.
    [brunnstroemRobust2024] J. Brunnström, M. Moonen, and F. Elvander, “Robust signal and noise covariance matrix estimation using Riemannian optimization,” presented at the European Signal Processing Conference (EUSIPCO), Sep. 2024
    """
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
    """Signal covariance estimator using the GEVD method.

    Parameters
    ----------
    Ry : ndarray of shape (dim, dim)
        Sample covariance matrix of the noisy signal.
    Rn : ndarray of shape (dim, dim)
        Sample covariance matrix of the noise.
    rank : int
        Rank of the signal covariance matrix. Must be less than or equal to dim.
    out : ndarray of shape (dim, dim), optional
        Output array for the estimated signal covariance matrix. If None, a new array is created.
    est_noise_cov : bool, optional
        If True, the noise covariance matrix is also estimated. Default is False.
    out_noise_cov : ndarray of shape (dim, dim), optional
        Output array for the estimated noise covariance matrix. If None, a new array is created.

    Returns
    -------
    Rx_hat : ndarray of shape (dim, dim)
        Estimated signal covariance matrix.
    Rv_hat : ndarray of shape (dim, dim)
        Estimated noise covariance matrix. Only returned if est_noise_cov is True.
    
    References
    ----------
    [serizelLowrank2014] R. Serizel, M. Moonen, B. V. Dijk, and J. Wouters, “Low-rank approximation based multichannel Wiener filter algorithms for noise reduction with application in cochlear implants,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 4, pp. 785–799, Apr. 2014, doi: 10.1109/TASLP.2014.2304240.
    [brunnstroemRobust2024] J. Brunnström, M. Moonen, and F. Elvander, “Robust signal and noise covariance matrix estimation using Riemannian optimization,” presented at the European Signal Processing Conference (EUSIPCO), Sep. 2024
    """
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
    """Generic function for estimating a signal and noise covariance matrix on a Riemannian manifold.

    Assumes that the signal covariance matrix is low-rank and the noise covariance matrix is full-rank. Specifically that
    the signal covariance matrix is returned in the form of X, where R_x = X @ X^H. 
    
    Parameters
    ----------
    ambient_dim : int
        Dimension of the covariance matrices.
    rank : int
        Rank of the signal covariance matrix.
    cost : function
        Cost function to minimize. Must be compatible with pymanopt.function.jax.
    manifold
        Riemannian manifold to optimize over. Must be compatible with pymanopt.manifolds.
    start_value : tuple of ndarrays, optional
        Initial value for the optimization. If None, the optimization starts from a random point.

    Returns
    -------
    Rx : ndarray of shape (ambient_dim, ambient_dim)
        Estimated signal covariance matrix.
    Rv : ndarray of shape (ambient_dim, ambient_dim)
        Estimated noise covariance matrix.
    """
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
    """Signal and noise covariance estimation using the Frobenius norm distance.
    
    Parameters
    ----------
    scm_noisy_signal : ndarray of shape (ambient_dim, ambient_dim)
        Sample covariance matrix of the noisy signal.
    scm_noise : ndarray of shape (ambient_dim, ambient_dim)
        Sample covariance matrix of the noise.
    rank : int
        Rank of the signal covariance matrix.
    start_value : tuple of ndarrays, optional
        Initial value for the optimization. If None, the optimization starts from a random point.
    
    Returns
    -------
    Rx : ndarray of shape (ambient_dim, ambient_dim)
        Estimated signal covariance matrix.
    Rv : ndarray of shape (ambient_dim, ambient_dim)
        Estimated noise covariance matrix.
    """
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
        return dist.frob_sq(scm_noisy_signal, Ry) + dist.frob_sq(scm_noise, Rv)
    
    return est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)

def est_manifold_wishart(scm_noisy_signal, scm_noise, rank, num_scm_samples, alpha=0.5, start_value=None):
    """Signal and noise covariance estimation using the Wishart log-likelihood distance.

    Parameters
    ----------
    scm_noisy_signal : ndarray of shape (ambient_dim, ambient_dim)
        Sample covariance matrix of the noisy signal.
    scm_noise : ndarray of shape (ambient_dim, ambient_dim)
        Sample covariance matrix of the noise.
    rank : int
        Rank of the signal covariance matrix.
    num_scm_samples : int
        Number of samples used to estimate the sample covariance matrix. Defines the degrees of freedom of the Wishart distribution.
    alpha : float, optional
        Weighting parameter between the noisy signal and the noise. Default is 0.5.
    start_value : tuple of ndarrays, optional
        Initial value for the optimization. If None, the optimization starts from a random point.

    Returns
    -------
    Rx : ndarray of shape (ambient_dim, ambient_dim)
        Estimated signal covariance matrix.
    Rv : ndarray of shape (ambient_dim, ambient_dim)
        Estimated noise covariance matrix.
    """
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
        
        llh_noisy_sig = dist.wishart_log_likelihood(scm_noisy_signal, Ry, num_scm_samples)
        llh_noise = dist.wishart_log_likelihood(scm_noise, Rv, num_scm_samples)

        return -alpha * llh_noisy_sig - (1-alpha) * llh_noise
    Rx_est, Rv_est = est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)

    return Rx_est, Rv_est





def est_manifold_frob_whitened(scm_noisy_signal, scm_noise, rank, alpha = 0.5, start_value=None):
    """Signal and noise covariance estimation using the Frobenius norm distance, with pre-whitening.
    
    This should be equivalent to est_gevd, but with the optimization done on a Riemannian manifold.

    Parameters
    ----------
    scm_noisy_signal : ndarray of shape (ambient_dim, ambient_dim)
        Sample covariance matrix of the noisy signal.
    scm_noise : ndarray of shape (ambient_dim, ambient_dim)
        Sample covariance matrix of the noise.
    rank : int
        Rank of the signal covariance matrix.
    alpha : float, optional
        Weighting parameter between the noisy signal and the noise. Default is 0.5. This parameter should
        not affect the optimal solution, but may affect the convergence of the optimization [vanRompaeyGEVD2018].

    Returns
    -------
    Rx : ndarray of shape (ambient_dim, ambient_dim)
        Estimated signal covariance matrix.
    Rv : ndarray of shape (ambient_dim, ambient_dim)
        Estimated noise covariance matrix.

    References
    ----------
    [serizelLowrank2014] R. Serizel, M. Moonen, B. V. Dijk, and J. Wouters, “Low-rank approximation based multichannel Wiener filter algorithms for noise reduction with application in cochlear implants,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 4, pp. 785–799, Apr. 2014, doi: 10.1109/TASLP.2014.2304240.
    [vanRompaeyGEVD2018] R. Van Rompaey and M. Moonen, “GEVD based speech and noise correlation matrix estimation for multichannel Wiener filter based noise reduction,” in 2018 26th European Signal Processing Conference (EUSIPCO), Sep. 2018, pp. 2544–2548. doi: 10.23919/EUSIPCO.2018.8553109.
    """
    ambient_dim = scm_noisy_signal.shape[0]
    scm_noisy_signal = (scm_noisy_signal + scm_noisy_signal.conj().T) / 2
    scm_noise = (scm_noise + scm_noise.conj().T) / 2

    # eigvec_invt is Q in the derivations, eigvec is Q^-H
    eigvals, eigvec = splin.eigh(scm_noisy_signal, scm_noise + 1e-10*np.eye(scm_noise.shape[0]))
    eigvec = np.flip(eigvec, axis=-1)
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
        return alpha * dist.frob_sq_weighted(scm_noisy_signal, Ry, Bv) + (1-alpha) * dist.frob_sq_weighted(scm_noise, Rv, Bv)
    return est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)


def est_manifold_frob_whitened_by_true_noise(scm_noisy_signal, scm_noise, rank, alpha=0.5, start_value=None):
    """Signal and noise covariance estimation using the Frobenius norm distance, with pre-whitening using the 'true' noise covariance matrix.

    This is equivalent to est_manifold_frob_whitened, except that the whitening parameter is extracted from 
    the optimization variable Rv instead of the sample covariance matrix. This should lead to the same solution, see [vanRompaeyGEVD2018].

    Parameters
    ----------
    scm_noisy_signal : ndarray of shape (ambient_dim, ambient_dim)
        Sample covariance matrix of the noisy signal.
    scm_noise : ndarray of shape (ambient_dim, ambient_dim)
        Sample covariance matrix of the noise.
    rank : int
        Rank of the signal covariance matrix.
    alpha : float, optional
        Weighting parameter between the noisy signal and the noise. Default is 0.5. This parameter should
        not affect the optimal solution, but may affect the convergence of the optimization [vanRompaeyGEVD2018].
    start_value : tuple of ndarrays, optional
        Initial value for the optimization. If None, the optimization starts from a random point.

    Returns
    -------
    Rx : ndarray of shape (ambient_dim, ambient_dim)
        Estimated signal covariance matrix.
    Rv : ndarray of shape (ambient_dim, ambient_dim)
        Estimated noise covariance matrix.

    References
    ----------
    [vanRompaeyGEVD2018] R. Van Rompaey and M. Moonen, “GEVD based speech and noise correlation matrix estimation for multichannel Wiener filter based noise reduction,” in 2018 26th European Signal Processing Conference (EUSIPCO), Sep. 2018, pp. 2544–2548. doi: 10.23919/EUSIPCO.2018.8553109.
    """
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

        eigvals, eigvec = matop.generalized_eigh(Ry, Rv + 1e-10*np.eye(scm_noise.shape[0]))
        eigvec = jnp.flip(eigvec, axis=-1)
        eigvals = jnp.flip(eigvals, axis=-1)
        Bv = eigvec.T.conj()

        return alpha * dist.frob_sq_weighted(scm_noisy_signal, Ry, Bv) + (1-alpha)*dist.frob_sq_weighted(scm_noise, Rv, Bv)
    return est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)


def est_manifold_airm(scm_noisy_signal, scm_noise, rank, alpha = 0.5, start_value=None):
    """Signal and noise covariance estimation using the affine-invariant Riemannian metric distance.
    
    Parameters
    ----------
    scm_noisy_signal : ndarray of shape (ambient_dim, ambient_dim)
        Sample covariance matrix of the noisy signal.
    scm_noise : ndarray of shape (ambient_dim, ambient_dim)
        Sample covariance matrix of the noise.
    rank : int
        Rank of the signal covariance matrix.
    alpha : float, optional
        Weighting parameter between the noisy signal and the noise. Default is 0.5.
    start_value : tuple of ndarrays, optional
        Initial value for the optimization. If None, the optimization starts from a random point.

    Returns
    -------
    Rx : ndarray of shape (ambient_dim, ambient_dim)
        Estimated signal covariance matrix.
    Rv : ndarray of shape (ambient_dim, ambient_dim)
        Estimated noise covariance matrix.
    """
    ambient_dim = scm_noisy_signal.shape[0]
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
        return alpha * dist.airm(Ry, scm_noisy_signal)**2 + (1 - alpha) * dist.airm(Rv, scm_noise)**2
    return est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)

def est_manifold_tylers_m_estimator(data_noisy_signal, data_noise, rank, start_value=None):
    """Signal and noise covariance estimation using the log-likelihood of an elliptical distribution.

    This is the proposed method of [brunnstroemRobust2024].

    Parameters
    ----------
    data_noisy_signal : ndarray of shape (ambient_dim, num_samples)
        Noisy signal samples.
    data_noise : ndarray of shape (ambient_dim, num_samples)
        Noise samples.
    rank : int
        Rank of the signal covariance matrix.
    start_value : tuple of ndarrays, optional
        Initial value for the optimization. If None, the optimization starts from a random point.

    Returns
    -------
    Rx : ndarray of shape (ambient_dim, ambient_dim)    
        Estimated signal covariance matrix.
    Rv : ndarray of shape (ambient_dim, ambient_dim)
        Estimated noise covariance matrix.

    References
    ----------
    [brunnstroemRobust2024] J. Brunnström, M. Moonen, and F. Elvander, “Robust signal and noise covariance matrix estimation using Riemannian optimization,” presented at the European Signal Processing Conference (EUSIPCO), Sep. 2024
    """
    ambient_dim = data_noisy_signal.shape[0]
    scaling = np.trace(scm(data_noisy_signal))
    
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

        llh_noisy_sig = dist.tyler_log_likelihood(data_noisy_signal, Ry)
        llh_noise = dist.tyler_log_likelihood(data_noise, Rv)
        return -llh_noisy_sig - llh_noise 
    Rx_est, Rv_est = est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)

    scaling = np.trace(scm(data_noisy_signal)) / np.trace(Rx_est + Rv_est)
    Rx_est *= scaling
    Rv_est *= scaling
    return Rx_est, Rv_est


def est_manifold_tylers_m_estimator_nonorm(data_noisy_signal, data_noise, rank, start_value=None):
    """Signal and noise covariance estimation using the log-likelihood of an elliptical distribution.

    This is a slight variation of the proposed method of [brunnstroemRobust2024], using a different definition of the log-likelihood.

    Parameters
    ----------
    data_noisy_signal : ndarray of shape (ambient_dim, num_samples)
        Noisy signal samples.
    data_noise : ndarray of shape (ambient_dim, num_samples)
        Noise samples.
    rank : int
        Rank of the signal covariance matrix.
    start_value : tuple of ndarrays, optional
        Initial value for the optimization. If None, the optimization starts from a random point.

    Returns
    -------
    Rx : ndarray of shape (ambient_dim, ambient_dim)    
        Estimated signal covariance matrix.
    Rv : ndarray of shape (ambient_dim, ambient_dim)
        Estimated noise covariance matrix.

    References
    ----------
    [brunnstroemRobust2024] J. Brunnström, M. Moonen, and F. Elvander, “Robust signal and noise covariance matrix estimation using Riemannian optimization,” presented at the European Signal Processing Conference (EUSIPCO), Sep. 2024
    """
    ambient_dim = data_noisy_signal.shape[0]

    trace_noisy_sig = np.trace(scm(data_noisy_signal))
    trace_noise = np.trace(scm(data_noise))
    trace_signal = trace_noisy_sig - trace_noise
    
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
        
        llh_noisy_sig = dist.tyler_log_likelihood_no_normalization(data_noisy_signal, Ry)
        llh_noise = dist.tyler_log_likelihood_no_normalization(data_noise, Rv)
        return -llh_noisy_sig - llh_noise 
    Rx_est, Rv_est = est_covariance_manifold(ambient_dim, rank, cost, joint_manifold, start_value=start_value)

    Rx_est = Rx_est / np.trace(Rx_est) * trace_signal
    Rv_est = Rv_est / np.trace(Rv_est) * trace_noise
    return Rx_est, Rv_est

def est_manifold_tylers_with_scale_opt(data_noisy_signal, data_noise, rank):
    """Signal and noise covariance estimation using the log-likelihood of an elliptical distribution, with scaling optimization.

    This is a slight variation of the proposed method of [brunnstroemRobust2024], which explicitly optimizes over a scaling factor.

    Parameters
    ----------
    data_noisy_signal : ndarray of shape (ambient_dim, num_samples)
        Noisy signal samples.
    data_noise : ndarray of shape (ambient_dim, num_samples)
        Noise samples.
    rank : int
        Rank of the signal covariance matrix.

    Returns
    -------
    Rx : ndarray of shape (ambient_dim, ambient_dim)    
        Estimated signal covariance matrix.
    Rv : ndarray of shape (ambient_dim, ambient_dim)
        Estimated noise covariance matrix.

    """
    ambient_dim = data_noisy_signal.shape[0]

    Ry_trace = np.trace(scm(data_noisy_signal))
    Rv_trace = np.trace(scm(data_noise))

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
        llh_noisy_sig = dist.tyler_log_likelihood(data_noisy_signal, Ry)
        llh_noise = dist.tyler_log_likelihood(data_noise, Rv)
        return -llh_noisy_sig - llh_noise + 1e3 * scaling_error
    
    problem = pymanopt.Problem(joint_manifold,cost,)
    optimizer = pymanopt.optimizers.TrustRegions(max_iterations=100, verbosity=2)
    cov_est = optimizer.run(problem, Delta_bar=8*np.sqrt(rank)).point
    X, Rv, scale = cov_est
    Rx = X @ jnp.conj(X.T)
    Rx *= scale[0] / jnp.trace(Rx)
    Rv *= scale[1] / jnp.trace(Rv)

    return Rx, Rv








