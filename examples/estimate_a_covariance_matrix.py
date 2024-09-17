"""Reproduces the first experiment in [brunnstromRobust2024].

References
----------
[brunnstromRobust2024] J. Brunnström, M. Moonen, and F. Elvander, “Robust signal and noise covariance matrix estimation using Riemannian optimization,” presented at the European Signal Processing Conference (EUSIPCO), Sep. 2024.
"""
import numpy as np
import pathlib

import json

import matplotlib.pyplot as plt

import aspcol.distance as aspdist
import aspcol.utilities as utils
import aspcol.plot as aspplot

import riecovest.covariance_estimation as covest
import riecovest.random_matrices as rm


def gen_sampled_data(dim, sig_rank, snr, num_cov_data, num_data, complex_data=True, noise_dist="gaussian", signal_dist="gaussian", rng=None, degrees_of_freedom=None, condition=1e-1):
    mean_zero = np.zeros((dim,))

    cov_signal, cov_noise = rm.random_signal_and_noise_covariance(dim, sig_rank, snr, complex_data=complex_data, rng=rng, condition=condition)
    #cov_noisy_sig = cov_noise + cov_signal

    if degrees_of_freedom is None:
            degrees_of_freedom = 2

    if noise_dist == "gaussian":
        if complex_data:
            cov_samples_noise_only = rm.sample_complex_gaussian(mean_zero, cov_noise, rng, num_cov_data)
            cov_samples_noise = rm.sample_complex_gaussian(mean_zero, cov_noise, rng, num_cov_data)
            noise = rm.sample_complex_gaussian(mean_zero, cov_noise, rng, num_data)
        else:
            cov_samples_noise_only = rm.sample_real_gaussian(mean_zero, cov_noise, rng, num_cov_data)
            cov_samples_noise = rm.sample_real_gaussian(mean_zero, cov_noise, rng, num_cov_data)
            noise = rm.sample_real_gaussian(mean_zero, cov_noise, rng, num_data)
    elif noise_dist == "t-distribution":
        if complex_data:
            cov_samples_noise_only = rm.sample_complex_t_distribution(mean_zero, cov_noise, rng, num_cov_data, degrees_of_freedom)
            cov_samples_noise = rm.sample_complex_t_distribution(mean_zero, cov_noise, rng, num_cov_data, degrees_of_freedom)
            noise = rm.sample_complex_t_distribution(mean_zero, cov_noise, rng, num_data, degrees_of_freedom)
        else:
            cov_samples_noise_only = rm.sample_real_t_distribution(mean_zero, cov_noise, rng, num_cov_data, degrees_of_freedom)
            cov_samples_noise = rm.sample_real_t_distribution(mean_zero, cov_noise, rng, num_cov_data, degrees_of_freedom)
            noise = rm.sample_real_t_distribution(mean_zero, cov_noise, rng, num_data, degrees_of_freedom)
    else:
        raise ValueError(f"Unknown noise distribution: {noise_dist}")
    
    if signal_dist == "gaussian":
        if complex_data:
            cov_samples_signal = rm.sample_complex_gaussian(mean_zero, cov_signal, rng, num_cov_data)
            signal = rm.sample_complex_gaussian(mean_zero, cov_signal, rng, num_data)
        else:
            cov_samples_signal = rm.sample_real_gaussian(mean_zero, cov_signal, rng, num_cov_data)
            signal = rm.sample_real_gaussian(mean_zero, cov_signal, rng, num_data)
    elif signal_dist == "t-distribution":
        if complex_data:
            cov_samples_signal = rm.sample_complex_t_distribution(mean_zero, cov_signal, rng, num_cov_data, degrees_of_freedom)
            signal = rm.sample_complex_t_distribution(mean_zero, cov_signal, rng, num_data, degrees_of_freedom)
        else:
            cov_samples_signal = rm.sample_real_t_distribution(mean_zero, cov_signal, rng, num_cov_data, degrees_of_freedom)
            signal = rm.sample_real_t_distribution(mean_zero, cov_signal, rng, num_data, degrees_of_freedom)
    else:
        raise ValueError(f"Unknown signal distribution: {signal_dist}")

    cov_samples_noisy_signal = cov_samples_signal + cov_samples_noise
    noisy_sig = noise + signal
    return noisy_sig, signal, noise, cov_samples_noisy_signal, cov_samples_noise_only, cov_signal, cov_noise


def exp(dim, rank, complex_data=True, noise_dist = "gaussian", signal_dist = "gaussian", rng=None, base_fig_folder=None, degrees_of_freedom=None):
    if rng is None:
        rng = np.random.default_rng(1234567)

    

    noisy_sig, signal, noise, cov_samples_noisy_signal, cov_samples_noise_only, cov_signal, cov_noise = gen_sampled_data(dim, rank, snr_lin, num_cov_data, num_data, complex_data, noise_dist, signal_dist, rng, degrees_of_freedom=degrees_of_freedom, condition=condition)


    # ========= SOLVE ESIMTATION PROBLEM BELOW HERE ============
    est_cov_sig, est_cov_noise = estimate_all(noisy_sig, signal, noise, cov_samples_noisy_signal, cov_samples_noise_only, rank, cov_signal, cov_noise)
    est_cov_sig, est_cov_noise, cov_signal, cov_noise = normalize_cov_matrices(est_cov_sig, est_cov_noise, cov_signal, cov_noise)
    
    all_cov_sig = {**est_cov_sig}
    all_cov_sig["true"] = cov_signal
    all_cov_noise = {**est_cov_noise}
    all_cov_noise["true"] = cov_noise

    show_matrices(all_cov_sig, fig_folder, "signal_covariance")
    show_matrices(all_cov_noise, fig_folder, "noise_covariance")




def estimate_all(cov_samples_noisy_signal, cov_samples_noise_only, true_sig_rank, true_signal_cov, true_noise_cov):
    scm_noisy_sig = covest.scm(cov_samples_noisy_signal)
    scm_noise = covest.scm(cov_samples_noise_only)
    tem_noisy_sig = covest.tyler_estimator(cov_samples_noisy_signal, 20)
    tem_noisy_sig = tem_noisy_sig / np.trace(tem_noisy_sig) * np.trace(scm_noisy_sig)
    tem_noise = covest.tyler_estimator(cov_samples_noise_only, 20)
    tem_noise = tem_noisy_sig / np.trace(tem_noise) * np.trace(scm_noise)

    cov_sig = {}
    cov_noise = {}

    cov_sig["subtract-scm"] = scm_noisy_sig - scm_noise
    cov_sig["gevd"] = covest.est_gevd(scm_noisy_sig, scm_noise, true_sig_rank)
    cov_sig["gevd tyler"] = covest.est_gevd(tem_noisy_sig, tem_noise, true_sig_rank)
    cov_sig["tyler"], cov_noise["tyler"] = covest.est_manifold_tylers_m_estimator(cov_samples_noisy_signal, cov_samples_noise_only, true_sig_rank)

    cov_noise["scm"] = scm_noise
    return cov_sig, cov_noise

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
    #aspplot.output_plot("pdf", fig_folder, f"matrices_{name}")




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
