import numpy as np
import scipy.stats as spstat


def low_rank_cholesky(A, rank):
    """Computes a low-rank decomposition of a positive definite matrix A.
    
    Returns L where A = L L*. Cholesky is a misnomer, since the decomposition returned here is not triangular.

    Parameters
    ----------
    A : ndarray of shape (dim, dim)
        Positive definite matrix to decompose.
    rank : int
        Rank of the low-rank decomposition.
    
    Returns
    -------
    L : ndarray of shape (dim, rank)
        Low-rank decomposition of A. Satisfies A = L L*.
    """
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.flip(eigvals, axis=-1)
    eigvecs = np.flip(eigvecs, axis=-1)

    L = eigvecs @ np.sqrt(np.diag(eigvals))
    L = L[:,:rank]

    if not np.allclose(L @ L.T.conj(), A):
        print(f"Warning: low rank cholesky does not reconstruct original matrix")
    return L 


def sample_elliptic_distribution(mean, covariance, rank, rng, num_samples):
    """Samples a central Elliptic distribution with given mean and scatter matrix. 
    
    See Theorem 3 of [ollilaComplex2012]
    
    Parameters
    ----------
    mean : complex ndarray of shape (dim,)
        Mean of the distribution.
    covariance : complex ndarray of shape (dim, dim)
        Covariance matrix of the distribution.
    rank : int
        Rank of the low-rank Cholesky factor of the covariance matrix.
    rng : numpy.random.Generator
        Random number generator.
    num_samples : int
        Number of samples to draw.

    Returns
    -------
    sample : ndarray of shape (dim, num_samples)
        Samples from the specified distribution.

    References
    ----------
    [ollilaComplex2012] E. Ollila, D. E. Tyler, V. Koivunen, and H. V. Poor, “Complex elliptically symmetric distributions: survey, new results and applications,” IEEE Transactions on Signal Processing, vol. 60, no. 11, pp. 5597–5625, Nov. 2012, doi: 10.1109/TSP.2012.2212433.
    """
    dim = covariance.shape[0]
    assert covariance.shape == (dim, dim)
    assert mean.shape == (dim,)
    A = low_rank_cholesky(covariance, rank)

    gaussian_vec = sample_complex_gaussian(np.zeros(dim), np.eye(dim), rng, num_samples)
    angular_vec = gaussian_vec / np.linalg.norm(gaussian_vec, axis=0, keepdims=True)

    elliptic_vec = mean[:,None] + scaling * A @ angular_vec
    
    raise NotImplementedError

def sample_complex_t_distribution(mean, covariance, rng, num_samples, degrees_of_freedom):
    """Sample from a complex t-distribution with given mean, covariance matrix and degrees of freedom.

    See equation (21) and then Section IV. A in [ollilaComplex2012]

    Parameters
    ----------
    mean : complex ndarray of shape (dim,)
        Mean of the complex Gaussian distribution.
    cov : complex ndarray of shape (dim, dim)
        Covariance matrix of the complex Gaussian distribution.
    rng : numpy.random.Generator
        Random number generator.
    num_samples : int
        Number of samples to draw.
    degrees_of_freedom : int
        Specifies which distribution from the t-family to sample from. Converges
        to Gaussian as degrees of freedom goes to infinity. 

    Returns
    -------
    sample : ndarray of shape (dim, num_samples)
        Samples from the specified distribution

    References
    ----------
    [ollilaComplex2012] E. Ollila, D. E. Tyler, V. Koivunen, and H. V. Poor, “Complex elliptically symmetric distributions: survey, new results and applications,” IEEE Transactions on Signal Processing, vol. 60, no. 11, pp. 5597–5625, Nov. 2012, doi: 10.1109/TSP.2012.2212433.
    """
    n = sample_complex_gaussian(np.zeros_like(mean), covariance, rng, num_samples)

    x =  rng.gamma(degrees_of_freedom / 2, 2, size=num_samples)
    texture = degrees_of_freedom / x

    scaled_normal = np.sqrt(texture)[None,:] * n
    sample = mean[:,None] + scaled_normal
    return sample


def real_gaussian_from_complex(mean, cov):
    """Converts mean and covariance of a complex Gaussian distribution to mean and covariance of a real Gaussian distribution.

    Use this to sample from a complex Gaussian distribution by sampling from the corresponding real Gaussian distribution.

    Parameters
    ----------
    mean : complex ndarray of shape (dim,)
        Mean of the complex Gaussian distribution.
    cov : complex ndarray of shape (dim, dim)
        Covariance matrix of the complex Gaussian distribution.
    
    Returns
    -------
    mean : ndarray of shape (2*dim,)
        Mean of the real Gaussian distribution.
    cov : ndarray of shape (2*dim, 2*dim)
        Covariance matrix of the real Gaussian distribution.
    """
    cov = 0.5 * np.block([[np.real(cov), -np.imag(cov)], [np.imag(cov), np.real(cov)]])
    mean = np.concatenate([np.real(mean), np.imag(mean)])
    return mean, cov

def sample_complex_gaussian(mean, cov, rng, num_samples):
    """Sample from a complex Gaussian distribution with given mean and covariance matrix.

    Parameters
    ----------
    mean : complex ndarray of shape (dim,)
        Mean of the complex Gaussian distribution.
    cov : complex ndarray of shape (dim, dim)
        Covariance matrix of the complex Gaussian distribution.
    rng : numpy.random.Generator
        Random number generator.
    num_samples : int
        Number of samples to draw.
    
    Returns
    -------
    sample : ndarray of shape (dim, num_samples)
        Complex Gaussian samples. 
    """
    joint_mean, joint_cov = real_gaussian_from_complex(mean, cov)
    sample = rng.multivariate_normal(joint_mean, joint_cov, size=num_samples).T
    return sample[:mean.shape[0],:] + 1j*sample[mean.shape[0]:,:]


def sample_real_t_distribution(mean, cov, rng, num_samples, degrees_of_freedom):
    """Sample from a real t-distribution with given mean, covariance matrix and degrees of freedom.

    Parameters
    ----------
    mean : ndarray of shape (dim,)
        Mean of the real Gaussian distribution.
    cov : ndarray of shape (dim, dim)
        Covariance matrix of the real Gaussian distribution.
    
    Returns
    -------
    sample : ndarray of shape (dim, num_samples)
        Real Gaussian samples.
    """
    var = spstat.multivariate_t(loc = mean, shape=cov, df=degrees_of_freedom, allow_singular=True).rvs(size=num_samples, random_state=rng).T
    return var

def sample_real_gaussian(mean, cov, rng, num_samples):
    """Sample from a real Gaussian distribution with given mean and covariance matrix.

    Parameters
    ----------
    mean : ndarray of shape (dim,)
        Mean of the real Gaussian distribution.
    cov : ndarray of shape (dim, dim)
        Covariance matrix of the real Gaussian distribution.
    
    Returns
    -------
    sample : ndarray of shape (dim, num_samples)
        Real Gaussian samples.
    """
    var = rng.multivariate_normal(mean, cov, size=num_samples).T
    return var




def random_signal_and_noise_covariance(dim, signal_rank, snr, complex_data=False, rng=None, condition = 1e-1):
    if complex_data:
        all_bases_noise = spstat.unitary_group.rvs(dim, random_state=rng)
        all_bases_signal = spstat.unitary_group.rvs(dim, random_state=rng)[:,:signal_rank]
    else:
        all_bases_noise = spstat.ortho_group.rvs(dim, random_state=rng)
        all_bases_signal = spstat.ortho_group.rvs(dim, random_state=rng)[:,:signal_rank]

    #ev_noise = rng.uniform(low = condition, high = 1, size = dim)
    #ev_signal = rng.uniform(low = condition, high = 1, size = signal_rank)
    ev_noise = spstat.loguniform.rvs(condition, 1, size=dim, random_state=rng)
    ev_signal = spstat.loguniform.rvs(condition, 1, size=signal_rank, random_state=rng)

    base_factors_noise =  np.diag(ev_noise)
    base_factors_signal = np.diag(ev_signal)

    cov_noise = all_bases_noise @ base_factors_noise @ all_bases_noise.conj().T
    cov_signal = all_bases_signal @ base_factors_signal @ all_bases_signal.conj().T

    cov_signal = cov_signal * snr / np.trace(cov_signal)
    cov_noise = cov_noise / np.trace(cov_noise)
    total_factor = dim / (np.trace(cov_signal) + np.trace(cov_noise))
    cov_signal = cov_signal * total_factor
    cov_noise = cov_noise * total_factor

    return cov_signal, cov_noise

def random_covariance(dim, rank, complex_data = False, rng=None):
    """
    
    Parameters
    ----------

    rank : int or "full"
    
    """
    if rng is None:
        rng = np.random.default_rng()

    if complex_data:
        basis = rng.uniform(-1, 1, size = (dim, rank)) + 1j*rng.uniform(-1, 1, size = (dim, rank))
        cov = basis @ basis.conj().T
        if dim > 1:
            U = spstat.unitary_group.rvs(dim, random_state=rng)
            cov = U @ cov @ U.conj().T
    else:
        raise NotImplementedError
    return cov

def random_signal_and_noise_covariance_old_heuristic(dim, signal_rank, snr, complex_data=False, rng=None):
    if complex_data:
        noise_basis = rng.uniform(-1, 1, size = (dim, 2*dim)) + 1j*rng.uniform(-1, 1, size = (dim, 2*dim))
        cov_noise = noise_basis @ noise_basis.conj().T / (2*dim)
        cov_noise += 1e-2 * np.eye(dim)

        U = spstat.unitary_group.rvs(dim, random_state=rng)
        cov_noise = U @ cov_noise @ U.conj().T

        signal_basis = rng.uniform(-1, 1, size = (dim, signal_rank)) + 1j*rng.uniform(-1, 1, size = (dim, signal_rank))
        cov_signal = signal_basis @ signal_basis.conj().T
        U = spstat.unitary_group.rvs(dim, random_state=rng)
        cov_signal = U @ cov_signal @ U.conj().T
    else:
        noise_basis = rng.uniform(-1, 1, size = (dim, 2*dim))
        cov_noise = noise_basis @ noise_basis.T / (2*dim)
        cov_noise += 1e-2 * np.eye(dim)

        U = spstat.ortho_group.rvs(dim, random_state=rng)
        cov_noise = U @ cov_noise @ U.T

        signal_basis = rng.uniform(-1, 1, size = (dim, signal_rank))
        cov_signal = signal_basis @ signal_basis.T
        U = spstat.ortho_group.rvs(dim, random_state=rng)
        cov_signal = U @ cov_signal @ U.T

    cov_signal = cov_signal * snr / np.trace(cov_signal)
    cov_noise = cov_noise / np.trace(cov_noise)
    total_factor = dim / (np.trace(cov_signal) + np.trace(cov_noise))
    cov_signal = cov_signal * total_factor
    cov_noise = cov_noise * total_factor

    return cov_signal, cov_noise




if __name__ == "__main__":
    rng=np.random.default_rng(12345678)
    num_samples = int(1e6)
    cov = random_covariance(1, 1, complex_data=True, rng=rng)
    mean = np.zeros((1,))
    gaussian = sample_complex_gaussian(mean, cov, rng, num_samples)
    t_dist_1000 = sample_complex_t_distribution(mean, cov, 1000, rng, num_samples)
    t_dist_5 = sample_complex_t_distribution(mean, cov, 5, rng, num_samples)

    num_bins = 1000
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,1, sharex=True)
    axes[0].hist(gaussian.real.T, num_bins, label="Gaussian", density=True, log=True)
    axes[1].hist(t_dist_1000.real.T, num_bins, label="t-dist 1000", density=True, log=True)
    axes[2].hist(t_dist_5.real.T, num_bins, label="t-dist 5", density=True, log=True)
    for ax in axes:
        ax.legend()
        ax.grid(True)
    plt.show()