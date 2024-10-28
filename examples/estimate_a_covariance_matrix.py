"""Reproduces the first experiment in [brunnstromRobust2024].

References
----------
[brunnstromRobust2024] J. Brunnström, M. Moonen, and F. Elvander, “Robust signal and noise covariance matrix estimation using Riemannian optimization,” presented at the European Signal Processing Conference (EUSIPCO), Sep. 2024.
"""
import numpy as np
import matplotlib.pyplot as plt

import riecovest.covariance_estimation as covest
import riecovest.random_matrices as rm

def normalize_cov_matrices(est_cov_sig, est_cov_noise, cov_signal, cov_noise):
    dim = cov_signal.shape[-1]

    cov_signal *= dim / np.trace(cov_signal + cov_noise)
    cov_noise *= dim / np.trace(cov_signal + cov_noise)

    for est_name, est_sig in est_cov_sig.items():
        if est_name in est_cov_noise:
            est_noise = est_cov_noise[est_name]
        else:
            est_noise = est_cov_noise["scm"]

        sum_trace = np.trace(est_sig + est_noise)
        est_cov_sig[est_name] = est_sig * dim / sum_trace
        est_cov_noise[est_name] = est_noise * dim / sum_trace
    return est_cov_sig, est_cov_noise, cov_signal, cov_noise

def show_matrices(mat_dict):
    fig, axes = plt.subplots(len(mat_dict), 3, figsize=(8, 3*len(mat_dict)))
    for i, (est_name, mat) in enumerate(mat_dict.items()):
        cax = axes[ i, 0].matshow(np.real(mat))
        fig.colorbar(cax)
        cax = axes[ i, 1].matshow(np.imag(mat))
        fig.colorbar(cax)
        cax = axes[ i, 2].matshow(np.abs(mat))
        fig.colorbar(cax)

        axes[i,0].set_title(f"Real: {est_name}")
        axes[i,1].set_title(f"Imag: {est_name}")
        axes[i,2].set_title(f"Abs: {est_name}")
        
    plt.show()


def main():
    dim = 8
    sig_rank = 2

    num_data = 12
    snr_db = 0
    snr = 10**(snr_db / 10)
    condition = 1e-2
    degrees_of_freedom = 3
    rng = np.random.default_rng(1234567)

    # Generate "true" covariance matrices
    mean_zero = np.zeros((dim,))
    cov_signal, cov_noise = rm.random_signal_and_noise_covariance(dim, sig_rank, snr, complex_data=True, rng=rng, condition=condition)

    # Sample t-distributed data
    samples_noise_only = rm.sample_complex_t_distribution(mean_zero, cov_noise, rng, num_data, degrees_of_freedom)
    samples_noise = rm.sample_complex_t_distribution(mean_zero, cov_noise, rng, num_data, degrees_of_freedom)
    samples_signal = rm.sample_complex_t_distribution(mean_zero, cov_signal, rng, num_data, degrees_of_freedom)
    samples_noisy_signal = samples_signal + samples_noise

    # Compute sample covariance matrices
    scm_noisy_sig = covest.scm(samples_noisy_signal)
    scm_noise = covest.scm(samples_noise_only)

    # Estimate covariance matrices
    cov_sig = {}
    cov_noise = {}
    print(f"Estimating using the subtraction method")
    cov_sig["subtract-scm"] = scm_noisy_sig - scm_noise
    print(f"Estimating using the GEVD method")
    cov_sig["gevd"] = covest.est_gevd(scm_noisy_sig, scm_noise, sig_rank)
    print(f"Estimating using the Robust Riemannian method")
    cov_sig["tyler"], cov_noise["tyler"] = covest.est_manifold_tylers_m_estimator(samples_noisy_signal, samples_noise_only, sig_rank)

    cov_noise["scm"] = scm_noise
    cov_sig["true"] = cov_signal
    cov_noise["true"] = cov_noise

    show_matrices(cov_sig)



if __name__ == "__main__":
    main()
