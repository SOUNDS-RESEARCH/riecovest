"""Reproduces the second experiment in [brunnstromRobust2024].

Uses the MeshRIR dataset, and it must be downloaded from the original source. The dataset is not included in the repository. Choose the correct path to the dataset in the code below, as well as putting the irutilities.py from the MeshRIR dataset in the same folder.

References
----------
[brunnstromRobust2024] J. Brunnström, M. Moonen, and F. Elvander, “Robust signal and noise covariance matrix estimation using Riemannian optimization,” presented at the European Signal Processing Conference (EUSIPCO), Sep. 2024.
"""
import numpy as np
import pandas as pd
import pathlib
import json
import sys
import soundfile as sf
import scipy.linalg as splin
import scipy.signal as spsig
import scipy.io as spio
import scipy.stats as spstat
import samplerate as sr_convert
from seaborn_qqplot import pplot

import matplotlib.pyplot as plt

import aspcol.distance as aspdist
import aspcol.filterclasses as fc
import aspcol.utilities as utils
import aspcol.plot as aspplot

import riecovest.covariance_estimation as covest
import riecovest.random_matrices as rm

import seaborn as sns

def analyze_audio_signals(signal_stft, noise_stft, fig_folder):
    num_freq = signal_stft.shape[0]
    dim = signal_stft.shape[1]

    Rx = np.stack([covest.scm(signal_stft[f,:,:]) for f  in range(num_freq)], axis=0)
    Rx_eigvals = np.flip(np.linalg.eigvalsh(Rx), axis=-1)

    Rv = np.stack([covest.scm(noise_stft[f,:,:]) for f  in range(num_freq)], axis=0)
    Rv_eigvals = np.flip(np.linalg.eigvalsh(Rv), axis=-1)

    fig, axes = plt.subplots(2,1)
    axes[0].set_title("Signal eigenvalues")
    axes[1].set_title("Noise eigenvalues")
    for i in range(3):
        axes[0].plot(Rx_eigvals[:,i], label=f"{i}th largest")
    for i in range(dim):
        axes[1].plot(Rv_eigvals[:,i], label=f"{i}th largest")
    for ax in axes:
        ax.legend()
        aspplot.set_basic_plot_look(ax)
    aspplot.output_plot("tikz", fig_folder, "covariance_eigenvalues", keep_only_latest_tikz=False)

    fig, axes = plt.subplots(2,1)
    axes[0].set_title("Signal eigenvalues")
    axes[1].set_title("Noise eigenvalues")
    for i in range(3):
        axes[0].plot(10*np.log((Rx_eigvals[:,i]+1e-10)**2), label=f"{i}th largest")
    for i in range(dim):
        axes[1].plot(10*np.log((Rv_eigvals[:,i]+1e-10)**2), label=f"{i}th largest")
    for ax in axes:
        ax.legend()
        aspplot.set_basic_plot_look(ax)
    aspplot.output_plot("tikz", fig_folder, "covariance_eigenvalues_db", keep_only_latest_tikz=False)

    num_plots = 2
    jump = num_freq//num_plots
    freq_idxs = np.arange(num_freq)[jump//2::jump]
    for f in freq_idxs:
        noise_seq = np.real(noise_stft[f, 0, :])
        qq_against_gaussian(noise_seq, fig_folder, f"_noise_freqidx_{f}")

        sig_seq = np.real(signal_stft[f, 0, :])
        qq_against_gaussian(sig_seq, fig_folder, f"_signal_freqidx_{f}")
    

    #t_dist_fit_over_freqs(noise_stft, fig_folder, "noise")
    #t_dist_fit_over_freqs(signal_stft, fig_folder, "signal")

def t_dist_fit_over_freqs(stft, fig_folder, plot_name= ""):
    df = np.zeros(stft.shape[0])
    mean = np.zeros(stft.shape[0])
    std = np.zeros(stft.shape[0])
    for f in range(stft.shape[0]):
        df[f], mean[f], std[f] = spstat.t.fit(np.real(stft[f,0,:]))

    fig, axes = plt.subplots(3 ,1, figsize =(10, 5))
    axes[0].plot(df)
    axes[1].plot(mean)
    axes[2].plot(std)
    
    axes[0].set_ylabel("Degrees of freedom")
    axes[1].set_ylabel("Mean")
    axes[2].set_ylabel("Scale (standard deviation)")
    for ax in axes:
        ax.set_xlabel("Frequency bin index")
        aspplot.set_basic_plot_look(ax)
    aspplot.output_plot("tikz", fig_folder, f"t_fit_over_freqs_{plot_name}", keep_only_latest_tikz=False)

def qq_against_gaussian(seq, fig_folder, plot_name=""):
    #noise_mean, noise_std = spstat.norm.fit(seq)
    gaussian_dist = spstat.norm() #spstat.norm(loc=noise_mean, scale=noise_std)
    
    plt_object = pplot(pd.DataFrame({"data" : seq}), x="data", y=gaussian_dist.dist, kind='qq', height=4, aspect=2, display_kws={"identity":True})
    
    aspplot.output_plot("tikz", fig_folder, f"qq_gaussian{plot_name}", keep_only_latest_tikz=False)

def resample_multichannel(ir, ratio):
    if ir.ndim == 3:
        assert ir.shape[0] == 1
        ir = ir[0,...]
    
    ir_out = []
    for i in range(ir.shape[0]):
        ir_out.append(sr_convert.resample(ir[i,...], ratio, "sinc_best").astype(float))
    ir_out = np.stack(ir_out, axis=0)
    return ir_out

def load_meshrir(sr):
    orig_sr = 48000
    ratio = sr / orig_sr

    #path_to_data_folder = pathlib.Path("c:/research/datasets/S32-M441_npy")

    pos_mic, pos_src, ir_hires = irutilities.loadIR(MESHRIR_FOLDER)
    pos_mic = -pos_mic
    ir_hires = ir_hires[(0,19,21),...]

    # x_lim = [-0.21, 0.21]
    # y_lim = [-0.21, 0.21]
    # z_lim = [-0.21, 0.21]
    # x_select = np.logical_and(x_lim[0] < pos_mic[:,0], pos_mic[:,0] < x_lim[1])
    # y_select = np.logical_and(y_lim[0] < pos_mic[:,1], pos_mic[:,1] < y_lim[1])
    # z_select = np.logical_and(z_lim[0] < pos_mic[:,2], pos_mic[:,2] < z_lim[1])
    # select = np.logical_and(np.logical_and(x_select, y_select), z_select)
    # ir_hires_selected = ir_hires[select,:]
    mic_idxs = np.array([1, 3, 19, 23, 25])
    ir_hires_selected = ir_hires[:,mic_idxs,:]
    pos_selected = pos_mic[mic_idxs,:]

    # pos_selected = pos_mic[select,:]
    ir = []
    for i in range(ir_hires.shape[0]):
        ir.append(resample_multichannel(ir_hires_selected[i,:,:], ratio))
    ir = np.stack(ir, axis=0)
    #mean_power = np.mean(np.sum(ir**2, axis=-1))
    #ir /= np.sqrt(mean_power)
    ir_sig = ir[0:1,:,:]
    ir_noise = ir[1:,:,:]

    return ir_sig, ir_noise, pos_selected

def load_single_audio():
    dataset_path = pathlib.Path("c:/research/datasets/LibriSpeech/dev-clean")
    for subpath in dataset_path.iterdir():
        if subpath.is_dir():
            for single_speaker_path in subpath.iterdir():
                if single_speaker_path.is_dir():
                    for audio_file_path in single_speaker_path.iterdir():
                        if audio_file_path.suffix == ".flac":
                            audio, sr = sf.read(audio_file_path)
                            audio = audio[15000:25000]
                            return audio, sr

def wola_batch_analysis(signal, block_size):
    """Constructs the entire WOLA spectrum for a signal
    
    Parameters
    ----------
    signal : ndarray of shape (num_channels, num_samples)
    sr : int
        samplerate
    block_size : int
        for now is required to be divisible by 2
    
    Returns
    -------
    wola_spec : ndarray of shape (num_freqs, num_channels, num_blocks)
    """
    assert block_size % 2 == 0
    assert signal.ndim == 2
    num_channels = signal.shape[0]
    num_samples = signal.shape[1]
    hop = block_size // 2
    wola = fc.WOLA(num_in=num_channels, num_out = 1, block_size=block_size, overlap=hop)
    num_freqs = wola.spectrum.shape[-1]

    num_blocks = num_samples // hop
    wola_spec = np.zeros((num_freqs, num_channels, num_blocks), complex)
    for i in range(num_blocks):
        wola.analysis(signal[:,i*hop:(i+1)*hop])
        wola_spec[...,i:i+1] = np.moveaxis(wola.spectrum, -1, 0)
    return wola_spec

def wola_batch_synthesis(spec, block_size):
    """Reconstructs the full signal from a set of time-frequency WOLA coefficients

    Parameters
    ----------
    spec : ndarray of shape (num_freq, num_channels, num_blocks)
        the WOLA coefficients, e.g. the values returned from wola_batch_analysis
    block_size : int
    
    Returns
    -------
    signal : ndarray of shape (num_channels, num_samples)
    """
    assert block_size % 2 == 0
    assert spec.ndim == 3
    num_freq = spec.shape[0]
    num_channels = spec.shape[1]
    num_blocks = spec.shape[2]
    hop = block_size // 2
    num_samples = hop * num_blocks

    wola = fc.WOLA(num_in=num_channels, num_out = 1, block_size=block_size, overlap=hop)
    signal = np.zeros((num_channels, num_samples))
    for i in range(num_blocks):
        wola.spectrum[...] = np.moveaxis(spec[...,i:i+1], 0, -1)
        signal[:,i*hop : (i+1)*hop] = np.squeeze(wola.synthesis(), 1)
    return signal

def load_impulsive_noise(dim, sr):
    noise_path = pathlib.Path(__file__).parent.joinpath("data").joinpath("handling_noise_long")
    noise_rec = []
    for i in range(1, dim+1):
        noise_single, sr_orig = sf.read(noise_path.joinpath(f"handling_noise_0{i}.wav"))
        noise_rec.append(noise_single)

    shortest_len = np.min([sig.shape[-1] for sig in noise_rec])
    noise_rec = [sig[:shortest_len] for sig in noise_rec]
    noise_audio = np.stack(noise_rec, axis=0)

    ratio = sr / sr_orig
    noise_audio = resample_multichannel(noise_audio, ratio)
    return noise_audio

def load_speech(sr):
    data_path = pathlib.Path(__file__).parent.joinpath("data").joinpath("speech_sig_cont")
    mat_file = spio.loadmat(str(data_path))
    audio = np.squeeze(mat_file["Speech_sig_cont"]).astype(float)
    speech_orig_sr = mat_file["Fs"][0,0]

    ratio = sr / speech_orig_sr

    speech_sig = resample_multichannel(audio[None,:], ratio)
    return speech_sig

def propagate(ir, sig):
    assert ir.ndim == 3
    assert ir.shape[0] == sig.shape[0]
    rir_len = ir.shape[-1]

    filt_sig = []
    for i in range(ir.shape[0]):
        filt_sig.append(spsig.fftconvolve(sig[i,None,:], ir[i,:,:], axes=-1))
    filt_sig = np.stack(filt_sig, axis=0)
    filt_sig = np.sum(filt_sig, axis=0)
    filt_sig = filt_sig[:,rir_len:-rir_len]
    return filt_sig

def gen_data(dim, sig_rank, snr, num_cov_data, num_data, fig_folder, noise_factor, freq_idx, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(noise_factor, (np.ndarray, list)):
        noise_factor = [noise_factor]

    num_blocks_per_segment = num_cov_data
    num_segments = 16
    block_size = 2048
    sr = 16000
    min_seconds = int(np.ceil((2 + 2 * num_segments) * num_blocks_per_segment * block_size / sr))
    sig_len_orig = sr * (min_seconds + 3)


    # ======== GENERATE TIME DOMAIN SIGNALS ============
    rir_signal, rir_noise, pos_mic = load_meshrir(sr)
    speech_sig = rm.sample_real_gaussian(np.zeros((1)), np.ones((1,1)), rng, sig_len_orig)
    noise_src = rng.normal(loc = 0, scale=1, size=(rir_noise.shape[0],speech_sig.shape[-1]))
    #speech_sig = load_speech(sr)
    #noise_factor = 10 # for speech
    #noise_factor = 1e4 # for gaussian
    #noise_impulsive = load_impulsive_noise(dim, sr) * noise_factor
    #noise_impulsive = np.tile(noise_impulsive, (1, 10))
    
    
    signal = propagate(rir_signal, speech_sig)
    noise_spatial = propagate(rir_noise, noise_src)
    sig_len = signal.shape[-1]

    scatter_noise = np.eye(dim)
    base_impulsive_noise = load_impulsive_noise(dim, sr)

    required_tile = 1 + sig_len // base_impulsive_noise.shape[-1]

    base_impulsive_noise = np.tile(base_impulsive_noise, (1,required_tile))
    base_impulsive_noise = base_impulsive_noise[:,:sig_len]
    #base_impulsive_noise = rm.sample_real_t_distribution(np.zeros((dim)), scatter_noise, rng, sig_len, degrees_of_freedom=1)
    noise_impulsive_all = [base_impulsive_noise * nf for nf in noise_factor]# *10

    # use *10 for degrees = 3
    # use *1 for degrees = 2
    # use 1e-2 for degrees = 1
     #* 1e-4
    #signal = signal[:,:noise_impulsive.shape[-1]]
    #if signal.shape[-1] < noise_impulsive.shape[-1]:
    #    noise_impulsive = noise_impulsive[:,:signal.shape[-1]]

    #signal /= np.sqrt(np.mean(signal**2))
    signal_power_mean = np.mean(signal**2)

    noise = [noise_spatial + ni for ni in noise_impulsive_all]

    noise_power_mean = [np.mean(ns**2) for ns in noise]
    mean_snr = [signal_power_mean / npm for npm in noise_power_mean]
    tdsig_info = {
        "mean SNR" : mean_snr,
        "samplerate" : sr,
        "total signal length" : sig_len,
        "signal power mean" : signal_power_mean,
        "impulsive noise power mean" : [np.mean(ni**2) for ni in noise_impulsive_all],
        "spatial noise variance" : np.mean(noise_spatial**2), 
        "total noise power mean" : noise_power_mean,
        "noise_factor" : noise_factor,
        #"noise_factor" : noise_factor
    }
    with open(fig_folder.joinpath("td_signal_info.json"), "w") as f:
        json.dump(tdsig_info, f, indent=4)


    fig, axes = plt.subplots(2 + len(noise), 1, figsize = (4 + 3*len(noise), 5), sharex=True)
    axes[0].plot(signal.T, alpha=0.7, label="signal")
    axes[1].plot(noise_spatial.T, alpha=0.7, label="noise gaussian")
    for i, ni in enumerate(noise_impulsive_all):
        axes[2+i].plot(ni.T, alpha=0.7, label=f"noise impulsive nf:{noise_factor[i]}")
    for ax in axes:
        aspplot.set_basic_plot_look(ax)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
    aspplot.output_plot("pdf", fig_folder, "time_domain_signals", keep_only_latest_tikz=False)

    # ========== COMBINED SIGNAL ============    
    noisy_sig = [signal + ns for ns in noise]

    freqs = np.fft.rfftfreq(block_size, 1 / sr)
    
    noise_stft = [wola_batch_analysis(ns, block_size) for ns in noise]
    signal_stft = wola_batch_analysis(signal, block_size)
    noisy_sig_stft = [wola_batch_analysis(ns, block_size) for ns in noisy_sig]

    plot_stft(signal_stft, "signal", fig_folder)
    [plot_stft(ns_stft, f"noise_nf{noise_factor[i]}", fig_folder) for i, ns_stft in enumerate(noise_stft)]
    [plot_stft(ns_stft, f"noisy_sig_nf{noise_factor[i]}", fig_folder) for i, ns_stft in enumerate(noisy_sig_stft)]

    #analyze_audio_signals(signal_stft, noise_stft, fig_folder)


    pass

    noise_stft = np.stack([ns[freq_idx,...] for ns in noise_stft], axis=0)
    noisy_sig_stft = np.stack([ns[freq_idx,...] for ns in noisy_sig_stft], axis=0)
    signal_stft = np.tile(signal_stft[freq_idx:freq_idx+1,...], (noise_stft.shape[0], 1,1))
    #noise_imp_stft = [wola_batch_analysis(ni, block_size) for ni in noise_impulsive_all]
    #plot_stft(noise_imp_stft, "noise_impulsive", fig_folder)


    num_freqs = signal_stft.shape[0]
    num_blocks = signal_stft.shape[-1]
    num_segments_total = num_blocks // num_blocks_per_segment
    #remainder = num_blocks % num_blocks_per_segment
    
    cov_samples_noise_only = np.zeros((num_segments_total, num_freqs, dim, num_blocks_per_segment), dtype=complex)
    cov_samples_signal_only = np.zeros((num_segments_total, num_freqs, dim, num_blocks_per_segment), dtype=complex)
    cov_samples_noisy_signal = np.zeros((num_segments_total, num_freqs, dim, num_blocks_per_segment), dtype=complex)
    offset = 2
    for i in range(num_segments_total):
        cov_samples_signal_only[i,...] = signal_stft[...,offset+i*num_blocks_per_segment:offset+(i+1)*num_blocks_per_segment]
        cov_samples_noisy_signal[i,...] = signal_stft[...,offset+i*num_blocks_per_segment:offset+(i+1)*num_blocks_per_segment] + \
                                            noise_stft[...,offset+i*num_blocks_per_segment:offset+(i+1)*num_blocks_per_segment]
        cov_samples_noise_only[i,...] = noise_stft[...,offset+i*num_blocks_per_segment:offset+(i+1)*num_blocks_per_segment]
    
    #num_segments = num_segments_total // 2
    cov_samples_noise_only = cov_samples_noise_only[num_segments:,...]
    cov_samples_signal_only = cov_samples_signal_only[:num_segments,...]
    cov_samples_noisy_signal = cov_samples_noisy_signal[:num_segments,...]
    if cov_samples_noise_only.shape[0] > num_segments:
        cov_samples_noise_only = cov_samples_noise_only[:-1,...]

    cov_signal = []
    cov_noise = []
    for i in range(num_segments):
        cov_signal.append(np.stack([covest.scm(cov_samples_signal_only[i,f,...]) for f in range(num_freqs)], axis=0))
        cov_noise.append(np.stack([covest.scm(cov_samples_noise_only[i,f,...]) for f in range(num_freqs)], axis=0))
    cov_signal = np.stack(cov_signal, axis=0)
    cov_noise = np.stack(cov_noise, axis=0)


    cov_samples_noise_only = cov_samples_noise_only[:num_segments,...]
    cov_samples_noisy_signal = cov_samples_noisy_signal[:num_segments,...]
    cov_signal = cov_signal[:num_segments,...]
    cov_noise = cov_noise[:num_segments,...]
   
    #cov_signal = np.stack([covest.scm(signal_stft[f,:,offset:offset+800]) for f in range(signal_stft.shape[0])], axis=0)
    #cov_noise = np.stack([covest.scm(noise_stft[f,:,offset:]) for f in range(signal_stft.shape[0])], axis=0)
    #cov_noise = np.zeros((freqs.shape[-1], dim, dim))
    #cov_noise[...] = np.eye(dim)[None,:,:] * signal_power_mean / snr
    

    #cov_samples_noisy_signal = cov_samples_signal + cov_samples_noise
    #noisy_sig = noise + signal
    return noisy_sig, signal, noise, cov_samples_noisy_signal, cov_samples_noise_only, cov_samples_signal_only, cov_signal, cov_noise, freqs



def plot_stft(spec, plot_name, fig_folder):
    assert spec.ndim == 3

    fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharey=True)
    clr = axes[0].imshow(np.abs(spec[:,0,:]))
    plt.colorbar(clr)
    clr = axes[1].imshow(10 * np.log10(np.abs(spec[:,0,:])**2))
    plt.colorbar(clr)


    axes[0].set_title("STFT Abs channel 0")
    axes[1].set_title("STFT Abs channel 0 (dB)")

    for ax in axes:
        ax.set_aspect("auto")
        #aspplot.set_basic_plot_look(ax)
        ax.set_xlabel("Time (block)")
        ax.set_ylabel("Frequency (index)")
    aspplot.output_plot("tikz", fig_folder, f"stft_{plot_name}", keep_only_latest_tikz=False)

def show_matrices(mat_dict, fig_folder, name = ""):
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
        
    
    aspplot.output_plot("pdf", fig_folder, f"matrices_{name}")

def show_eigenvalues(mat_dict, fig_folder, name = ""):

    fig, ax = plt.subplots(1,1, figsize=(8, 3))
    for i, (est_name, mat) in enumerate(mat_dict.items()):
        eigvals = np.linalg.eigvalsh(mat)
        ax.plot(eigvals, "-o", label=est_name)
        ax.set_xlabel("Eigenvalue index")
        ax.legend()

        aspplot.set_basic_plot_look(ax)
    aspplot.output_plot("pdf", fig_folder, f"eigenvalues_{name}")
    
def estimation_errors_all(cov_sig, cov_noise, true_signal_cov, true_noise_cov, fig_folder, plot_name=""):
    true_noisy_sig_cov = true_signal_cov + true_noise_cov

    estimation_errors(cov_sig, true_signal_cov, False, fig_folder, f"signal_covariance{plot_name}")
    estimation_errors(cov_noise, true_noise_cov, True, fig_folder, f"noise_covariance{plot_name}")

    cov_noisy_sig = {}
    for name, Rx in cov_sig.items():
        if name in cov_noise:
            Rv = cov_noise[name]
        else:
            Rv = cov_noise["scm"]
        cov_noisy_sig[name] = Rx + Rv
        #Ry_est = Rx + Rv
    estimation_errors(cov_noisy_sig, true_noisy_sig_cov, True, fig_folder, f"noisy_signal_covariance{plot_name}")

def estimation_errors(est, cov_true, full_rank, fig_folder, name):
    metrics = {}
    metrics["mse"] = {}
    metrics["nmse"] = {}
    metrics["mse trace-normalized"] = {}
    metrics["nmse trace-normalized"] = {}
    metrics["corrmat_distance"] = {}
    if full_rank:
        metrics["airm"] = {}
        metrics["kl divergence"] = {}
        
    for est_name, cov_est in est.items():
        metrics["mse"][est_name] = np.linalg.norm(cov_est - cov_true, ord="fro")**2 
        metrics["nmse"][est_name] = metrics["mse"][est_name] / np.linalg.norm(cov_true, ord="fro")**2
        #metrics["nmse (dB)"][est_name] = 10 * np.log10(metrics["nmse"][est_name])
        
        cov_est_trace_normalized = cov_est * (np.trace(cov_true) / np.trace(cov_est))
        metrics["mse trace-normalized"][est_name] = np.linalg.norm(cov_est_trace_normalized - cov_true, ord="fro")**2
        metrics["nmse trace-normalized"][est_name] = np.linalg.norm(cov_est_trace_normalized - cov_true, ord="fro")**2 / np.linalg.norm(cov_true, ord="fro")**2
        #metrics["nmse trace-normalized (dB)"][est_name] = 10 * np.log10(metrics["nmse trace-normalized"][est_name])
        metrics["corrmat_distance"][est_name] = aspdist.corr_matrix_distance(cov_est, cov_true).tolist()

        if full_rank:
            metrics["airm"][est_name] = covest.airm(cov_est, cov_true).tolist()
            metrics["kl divergence"][est_name] = aspdist.covariance_distance_kl_divergence(cov_est, cov_true).tolist()

    metrics_complex= {metric_name : {} for metric_name in metrics.keys()}
    for metric_name, metric_value in metrics.items():
        for est_name, est_value in metric_value.items():
            metrics[metric_name][est_name] = np.abs(est_value)
            metrics_complex[metric_name][f"{est_name} : real"] = np.real(est_value)
            metrics_complex[metric_name][f"{est_name} : imag"] = np.imag(est_value)

    with open(fig_folder.joinpath(f"estimation_errors_{name}.json"), "w") as f:
        json.dump(metrics, f, indent = 4)

    with open(fig_folder.joinpath(f"estimation_errors_complex_{name}.json"), "w") as f:
        json.dump(metrics_complex, f, indent = 4)

    metrics_db = {}
    for metric_name, metric_value in metrics.items():
        metrics_db[metric_name] = {k:10*np.log10(v) for k,v in metric_value.items()}
    with open(fig_folder.joinpath(f"estimation_errors_{name}_db.json"), "w") as f:
        json.dump(metrics_db, f, indent = 4)





def exp(noise_stft, signal_stft, noisy_sig_stft, cov_samples_noisy_signal, cov_samples_noise_only, cov_samples_signal_only, cov_signal, cov_noise, freqs, dim, rank, rng=None, base_fig_folder=None):
    if rng is None:
        rng = np.random.default_rng(1234567)

    #num_blocks = cov_samples_noisy_signal.shape[0]
    #num_freqs = cov_samples_noisy_signal.shape[1]
    dim = cov_samples_noisy_signal.shape[0]
    num_cov_data = cov_samples_noisy_signal.shape[1]
    
    if base_fig_folder is None:
        base_fig_folder = pathlib.Path(__file__).parent.joinpath("figs")
        base_fig_folder.mkdir(exist_ok=True)

    fig_folder = utils.get_unique_folder("figs_", base_fig_folder)
    fig_folder.mkdir()

    sim_info = {}
    #sim_info[f"Noise power"] = np.mean(np.abs(noise)**2)
    #sim_info[f"Signal power"] = np.mean(np.abs(signal)**2)
    #sim_info[f"Prior SNR (dB)"] = 10 * np.log10(np.mean(np.abs(signal)**2) / np.mean(np.abs(noise)**2))
    sim_info[f"Data points for cov estimation"] = num_cov_data
    sim_info[f"Dimension"] = dim
    sim_info[f"Signal rank"] = rank
    sim_info[f"complex data"] = True
    with open(fig_folder.joinpath("sim_info.json"),"w") as f:
        json.dump(sim_info, f, indent=4)

    # ========= SOLVE ESTIMATION PROBLEM BELOW HERE ============

    est_cov_sig, est_cov_noise = est_all_covariances(cov_samples_noisy_signal, cov_samples_noise_only, rank)
    est_cov_sig, est_cov_noise, cov_signal, cov_noise = normalize_cov_matrices(est_cov_sig, est_cov_noise, cov_signal, cov_noise)
    
    all_cov_sig = {**est_cov_sig}
    all_cov_sig["true"] = cov_signal
    all_cov_noise = {**est_cov_noise}
    all_cov_noise["true"] = cov_noise
    
    #estimation_errors(est_cov_sig, cov_signal, fig_folder, "signal_covariance")
    #estimation_errors(est_cov_noise, cov_noise, fig_folder, "noise_covariance")
    estimation_errors_all(est_cov_sig, est_cov_noise, cov_signal, cov_noise, fig_folder, plot_name=f"")
    
    np.savez(fig_folder.joinpath("cov_sig"),  **all_cov_sig)
    np.savez(fig_folder.joinpath("cov_noise"),  **all_cov_noise)
    faster_exp = True
    if not faster_exp:
        show_matrices(all_cov_sig, fig_folder, f"signal_covariance")
        show_matrices(all_cov_noise, fig_folder, f"noise_covariance")
        #show_rank(est_cov_sig, fig_folder, "signal_covariance")
        show_eigenvalues(all_cov_sig, fig_folder, f"signal")
        show_eigenvalues(all_cov_noise, fig_folder, f"noise")

    exp_spatial_filtering_single_freq(est_cov_sig, est_cov_noise, cov_samples_noisy_signal, cov_samples_noise_only, cov_samples_signal_only, fig_folder)


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


def exp_spatial_filtering_single_freq(est_cov_sig, est_cov_noise, noisy_sig_stft, noise_stft, signal_stft, fig_folder):
    mwf = {}
    noisy_sig_processed = {}
    noise_processed = {}
    signal_processed = {}
    for est_name, est_sig in est_cov_sig.items():
        if est_name in est_cov_noise:
            est_noise = est_cov_noise[est_name]
        else:
            est_noise = est_cov_noise["scm"]
        #The following should be equal
        #a = splin.solve((est_sig + est_noise).T, est_sig.T).T
        #b = est_sig @ splin.inv(est_sig + est_noise)
        W = splin.solve((est_sig + est_noise).T, est_sig.T).T
        if np.isnan(np.sum(W)):
            print(f"nan found")
        #W *= W.shape[-1] / np.trace(W)
        mwf[est_name] = W
        noisy_sig_processed[est_name] = W @ noisy_sig_stft
        noise_processed[est_name] = W @ noise_stft
        signal_processed[est_name] = W @ signal_stft
    
    show_matrices(mwf, fig_folder, name="mwf")

    pre_noise_power = np.mean(np.abs(noise_stft)**2)
    pre_sig_power = np.mean(np.abs(signal_stft)**2)
    metrics = {}
    metrics["post-snr"] = {}
    metrics["pre-snr"] = {}
    metrics["snr-ratio"] = {}
    metrics["post-signal-power"] = {}
    metrics["post-noise-power"] = {}
    metrics["nmse"] = {}
    metrics["signal-distortion"] = {}
    metrics["do-nothing-nmse"] = {}

    for est_name in mwf:
        noise_power = np.mean(np.abs(noise_processed[est_name])**2)
        sig_power = np.mean(np.abs(signal_processed[est_name])**2)
        metrics["pre-snr"][est_name] = pre_sig_power / pre_noise_power
        metrics["post-snr"][est_name] = sig_power / noise_power
        metrics["snr-ratio"][est_name] = metrics["post-snr"][est_name] / metrics["pre-snr"][est_name]
        metrics["post-noise-power"][est_name] = noise_power
        metrics["post-signal-power"][est_name] = sig_power

        metrics["do-nothing-nmse"][est_name] = 1 / metrics["pre-snr"][est_name]

        metrics["nmse"][est_name] = np.mean(np.abs(noisy_sig_processed[est_name] - signal_stft)**2) / np.mean(np.abs(signal_stft)**2)
        metrics["signal-distortion"][est_name] = np.mean(np.abs(signal_processed[est_name] - signal_stft)**2) / np.mean(np.abs(signal_stft)**2)

    with open(fig_folder.joinpath(f"eval_spatial_filtering.json"), "w") as f:
        json.dump(metrics, f, indent = 4)

    metrics_db = {}
    for metric_name, metric_value in metrics.items():
        metrics_db[metric_name] = {k:10*np.log10(v) for k,v in metric_value.items()}
    with open(fig_folder.joinpath(f"eval_spatial_filtering_db.json"), "w") as f:
        json.dump(metrics_db, f, indent = 4)
    #fig, axes = plt.subplots(3,len(mwf), figsize=(15, 5 * len(mwf)))




def est_all_covariances(cov_samples_noisy_signal, cov_samples_noise_only, true_sig_rank):
    scm_noisy_sig = covest.scm(cov_samples_noisy_signal)
    scm_noise = covest.scm(cov_samples_noise_only)
    tem_noisy_sig = covest.tyler_estimator(cov_samples_noisy_signal, 20)
    tem_noisy_sig = tem_noisy_sig * np.trace(scm_noisy_sig) / np.trace(tem_noisy_sig) 
    tem_noise = covest.tyler_estimator(cov_samples_noise_only, 20)
    tem_noise = tem_noisy_sig * np.trace(scm_noise) / np.trace(tem_noise) 

    cov_sig = {}
    cov_noise = {}

    cov_sig["subtract-scm"] = scm_noisy_sig - scm_noise
    cov_sig["gevd"], cov_noise["gevd"] = covest.est_gevd(scm_noisy_sig, scm_noise, true_sig_rank, est_noise_cov=True)
    cov_sig["gevd tyler"], cov_noise["gevd tyler"] = covest.est_gevd(tem_noisy_sig, tem_noise, true_sig_rank, est_noise_cov=True)

    cov_sig["tyler"], cov_noise["tyler"] = covest.est_manifold_tylers_m_estimator(cov_samples_noisy_signal, cov_samples_noise_only, true_sig_rank)#, 
    cov_noise["scm"] = scm_noise
    return cov_sig, cov_noise


def run_single_freq_exp(noise_stft, signal_stft, noisy_sig_stft,  cov_samples_noisy_signal, cov_samples_noise_only, cov_samples_signal_only, cov_signal, cov_noise, freqs, dim, rank, rng, base_fig_folder=None):
    if base_fig_folder is None:
        base_fig_folder = pathlib.Path(__file__).parent.joinpath("figs")
        base_fig_folder.mkdir(exist_ok=True)
    fig_folder = utils.get_unique_folder("figs_mc_", base_fig_folder)
    fig_folder.mkdir()

    num_segments = cov_samples_noisy_signal.shape[0]

    for i in range(num_segments):
        exp(noise_stft, signal_stft, noisy_sig_stft, cov_samples_noisy_signal[i,...], cov_samples_noise_only[i,...], cov_samples_signal_only[i,...], cov_signal[i,...], cov_noise[i,...], freqs, dim, rank, rng=rng, base_fig_folder=fig_folder)

    return fig_folder







def analyze_monte_carlo_trial(base_fig_folder):
    cov_sig_all = []
    cov_noise_all = []
    sim_info_all = []
    est_errors_all = []

    for fdr in base_fig_folder.iterdir():
        if fdr.is_dir():
            #if fdr.is_dir():
            #    fig_folder = fdr
            #    break
            cov_sig = np.load(fdr.joinpath("cov_sig.npz"))
            cov_sig = {k:v for k,v in cov_sig.items()}
            cov_sig_all.append(cov_sig)
            cov_noise = np.load(fdr.joinpath("cov_noise.npz"))
            cov_noise = {k:v for k,v in cov_noise.items()}
            cov_noise_all.append(cov_noise)
            with open(fdr.joinpath("sim_info.json"), "r") as f:
                sim_info = json.load(f)
            sim_info_all.append(sim_info)

            with open(fdr.joinpath("estimation_errors_signal_covariance.json"), "r") as f:
                est_errors = json.load(f)
                est_errors = {f"covsig - {key}" : val for key, val in est_errors.items()}
                est_errors_all.append(est_errors)

            with open(fdr.joinpath("estimation_errors_noise_covariance.json"), "r") as f:
                est_errors = json.load(f)
                est_errors = {f"covnoise - {key}" : val for key, val in est_errors.items()}
                est_errors_all.append(est_errors)
        
            with open(fdr.joinpath("estimation_errors_noisy_signal_covariance.json"), "r") as f:
                est_errors = json.load(f)
                est_errors = {f"covnoisysig - {key}" : val for key, val in est_errors.items()}
                est_errors_all.append(est_errors)

            with open(fdr.joinpath("eval_spatial_filtering.json"), "r") as f:
                est_errors = json.load(f)
                est_errors = {f"spatialfiltering - {key}" : val for key, val in est_errors.items()}
                est_errors_all.append(est_errors)

    summarize_metrics(est_errors_all, base_fig_folder)


def summarize_metrics(metric_list, fig_folder):
    metrics = {}
    for metric in metric_list:
        for est_name, est_metric in metric.items():
            if est_name not in metrics:
                metrics[est_name] = {}
            for metric_name, metric_value in est_metric.items():
                if not metric_name.endswith("(dB)"):
                    if metric_name not in metrics[est_name]:
                        metrics[est_name][metric_name] = []
                    metrics[est_name][metric_name].append(metric_value)

    metric_summary = {}

    for est_name, est_metric in metrics.items():
        if est_name not in metric_summary:
            metric_summary[est_name] = {}
        for metric_name, metric_value in est_metric.items():
            metric_summary[est_name][f"{metric_name} : sample mean"] = np.mean(metric_value)
            metric_summary[est_name][f"{metric_name} : sample variance"] = np.var(metric_value)
            metric_summary[est_name][f"{metric_name} : variance of sample mean"] = metric_summary[est_name][f"{metric_name} : sample variance"]/ len(metric_value)
            #metrics[name]["mse-mean"] = np.mean(mse)
            #metrics[name]["mse-variance"] = np.var(mse)
            #metrics[name]["mse-variance-of-sample-mean"] = metrics[name]["mse-variance"] / len(est)
    with open(fig_folder.joinpath("summary.json"), "w") as f:
        json.dump(metric_summary, f, indent=4)
    
    metrics_db = {}
    for est_name, est_metric in metric_summary.items():
        if est_name not in metrics_db:
            metrics_db[est_name] = {} 
        for metric_name, metric_value in est_metric.items():
            metrics_db[est_name][metric_name] = 10*np.log10(metric_value)
    with open(fig_folder.joinpath(f"summary_db.json"), "w") as f:
        json.dump(metrics_db, f, indent = 4)

    fig, ax = plt.subplots(1,1)
    for est_name, est_val in metrics["spatialfiltering - nmse"].items():
        ax.plot(10 * np.log10(np.array(est_val)**2), label=est_name)
    ax.legend()
    ax.set_xlabel("Time segment")
    ax.set_ylabel("NMSE (dB)")
    aspplot.set_basic_plot_look(ax)
    aspplot.output_plot("tikz", fig_folder, f"nmse_over_time", keep_only_latest_tikz=False)

def plot_parameter_exp(fig_folder):
    folders = []
    for fdr in fig_folder.iterdir():
        if fdr.is_dir():
            folders.append(fdr)

    with open(fig_folder.joinpath("parameter.json")) as f:
        parameter = json.load(f)
        parameter_name = list(parameter.keys())[0]
        parameter_vals = parameter[parameter_name]

    for fdr in folders:
        if fdr.stem.startswith("figs_"):
            analyze_monte_carlo_trial(fdr)
        
    val = {}
    summaries = []
    for fdr in folders:
        if fdr.stem.startswith("figs_"):
            with open(fdr.joinpath("summary.json")) as f:
                summary = json.load(f)
                summaries.append(summary)
    total_summary = list_of_dicts_to_dict_of_lists(summaries)     
    algo_names = get_all_algo_names(total_summary)

    for sum_name, sum_dict in total_summary.items():
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        #if np.isnan(np.sum(sum_dict[f"{nm} : sample mean"])) or sum_dict[f"{nm} : variance of sample mean"]:


        for nm in algo_names:
            if np.isnan(np.sum(sum_dict[f"{nm} : sample mean"])):
                pass
            else:
                ax.plot(parameter_vals, sum_dict[f"{nm} : sample mean"], "-x", label={nm})
        
        y_lim = ax.get_ylim()
        for i, nm in enumerate(algo_names):
            if np.isnan(np.sum(sum_dict[f"{nm} : sample mean"])) or np.isnan(np.sum(sum_dict[f"{nm} : variance of sample mean"])):
                print(f"Some Nan in {nm} for {sum_name}, so not everything is plotted")
            else:
                center = np.array(sum_dict[f"{nm} : sample mean"])
                width = np.sqrt(np.array(sum_dict[f"{nm} : variance of sample mean"]))
                if np.abs(np.sum(width)) > 0:
                    ax.fill_between(np.array(parameter_vals), center-width, center+width, alpha=0.4, color=f"C{i}")
        ax.set_ylim(y_lim)

        ax.legend()
        ax.set_xlabel(parameter_name)
        ax.set_ylabel(sum_name)
        aspplot.set_basic_plot_look(ax)
        aspplot.output_plot("tikz", fig_folder, f"{sum_name}_{parameter_name}", keep_only_latest_tikz=False)

        fig, ax = plt.subplots(1,1, figsize=(8,6))
        for nm in algo_names:
            if np.isnan(np.sum(sum_dict[f"{nm} : sample mean"])):
                pass
            else:
                ax.plot(parameter_vals, 10 * np.log10(sum_dict[f"{nm} : sample mean"]), "-x", label={nm})

        y_lim = ax.get_ylim()
        for i, nm in enumerate(algo_names):
            if np.isnan(np.sum(sum_dict[f"{nm} : sample mean"])) or np.isnan(np.sum(sum_dict[f"{nm} : variance of sample mean"])):
                print(f"Some Nan in {nm} for {sum_name}, so not everything is plotted")
            else:
                center = np.array(sum_dict[f"{nm} : sample mean"])
                width =  np.sqrt(np.array(sum_dict[f"{nm} : variance of sample mean"]))
                if np.abs(np.sum(width)) > 0:
                    lower_lim = center-width
                    lower_lim[lower_lim <= 0] = 1e-10
                    upper_lim = center + width
                    ax.fill_between(np.array(parameter_vals), 10*np.log10(lower_lim), 10*np.log10(upper_lim), alpha=0.4, color=f"C{i}")
        ax.set_ylim(y_lim)

        ax.legend()
        ax.set_xlabel(f"{parameter_name}")
        ax.set_ylabel(f"{sum_name} (dB)")
        aspplot.set_basic_plot_look(ax)
        aspplot.output_plot("tikz", fig_folder, f"{sum_name}_{parameter_name}_db", keep_only_latest_tikz=False)

        if np.min(parameter_vals) > 0:
            log_param_vals = np.log10(parameter_vals)
            fig, ax = plt.subplots(1,1, figsize=(8,6))
            for nm in algo_names:
                if np.isnan(np.sum(sum_dict[f"{nm} : sample mean"])):
                    pass
                else:
                    ax.plot(log_param_vals, 10 * np.log10(sum_dict[f"{nm} : sample mean"]), "-x", label={nm})

            y_lim = ax.get_ylim()
            for i, nm in enumerate(algo_names):
                if np.isnan(np.sum(sum_dict[f"{nm} : sample mean"])) or np.isnan(np.sum(sum_dict[f"{nm} : variance of sample mean"])):
                    print(f"Some Nan in {nm} for {sum_name}, so not everything is plotted")
                else:
                    center = np.array(sum_dict[f"{nm} : sample mean"])
                    width =  np.sqrt(np.array(sum_dict[f"{nm} : variance of sample mean"]))
                    if np.abs(np.sum(width)) > 0:
                        lower_lim = center-width
                        lower_lim[lower_lim <= 0] = 1e-10
                        upper_lim = center + width
                        ax.fill_between(np.array(log_param_vals), 10*np.log10(lower_lim), 10*np.log10(upper_lim), alpha=0.4, color=f"C{i}")
            ax.set_ylim(y_lim)

            ax.legend()
            ax.set_xlabel(f"{parameter_name} (log10)")
            ax.set_ylabel(f"{sum_name} (dB)")
            aspplot.set_basic_plot_look(ax)
            aspplot.output_plot("tikz", fig_folder, f"{sum_name}_{parameter_name}_logdb", keep_only_latest_tikz=False)

def get_all_algo_names(total_summary):
    one_dict = total_summary[list(total_summary.keys())[0]]

    algo_names = []
    for name in one_dict.keys():
        one_algo_name, _ = name.split(":")
        one_algo_name = one_algo_name.strip()
        if one_algo_name not in algo_names:
            algo_names.append(one_algo_name)
    return algo_names

def list_of_dicts_to_dict_of_lists(dct):
    new_dict = {}
    for name, sub_dict in dct[0].items():
        new_dict[name] = {}
        for sub_name, val in sub_dict.items():
            new_dict[name][sub_name] = []
    

    for name, sub_dict in new_dict.items():
        for sub_name, val in sub_dict.items():
            for i in range(len(dct)):
                new_dict[name][sub_name].append(dct[i][name][sub_name])
    return new_dict


def downsample_freqs(ds_factor, cov_samples_noisy_signal, cov_samples_noise_only, cov_samples_signal_only, cov_signal, cov_noise, freqs):
    num_freqs = freqs.shape[0]
    freq_idxs = np.arange(num_freqs)[ds_factor//2::ds_factor]

    cov_samples_noisy_signal = cov_samples_noisy_signal[:,freq_idxs,...]
    cov_samples_noise_only = cov_samples_noise_only[:,freq_idxs,...]
    cov_samples_signal_only = cov_samples_signal_only[:,freq_idxs,...]
    cov_signal = cov_signal[:,freq_idxs,...]
    cov_noise = cov_noise[:,freq_idxs,...]
    freqs = freqs[freq_idxs,...]

    return cov_samples_noisy_signal, cov_samples_noise_only, cov_samples_signal_only, cov_signal, cov_noise, freqs


def run_full_speech_exp():
    fig_folder = utils.get_unique_folder("figs_", BASE_FIG_FOLDER)
    fig_folder.mkdir()

    num_cov_data = 64
    num_data = 1000
    snr_db = 10
    snr_lin = 10**(snr_db / 10)
    dim = 5
    rank = 1

    rng = np.random.default_rng(1234564354)

    #noise_factor = np.logspace(-5, -1, 9).tolist()
    noise_factor = np.logspace(1, 5, 9).tolist()
    #noise_factor = [10**(3), 10**(3), 10**(4), 10**(5)]
    freq_idx = 64
    noisy_sig, signal, noise, cov_samples_noisy_signal, cov_samples_noise_only, cov_samples_signal_only, cov_signal, cov_noise, freqs = gen_data(dim, rank, snr_lin, num_cov_data, num_data, fig_folder, noise_factor, freq_idx, rng)

    
    # cov_samples_noisy_signal = cov_samples_noisy_signal[:,freq_idx:freq_idx+1,...]
    # cov_samples_noise_only = cov_samples_noise_only[:,freq_idx:freq_idx+1,...]
    # cov_samples_signal_only = cov_samples_signal_only[:,freq_idx:freq_idx+1,...]
    # cov_signal = cov_signal[:,freq_idx:freq_idx+1,...]
    # cov_noise = cov_noise[:,freq_idx:freq_idx+1,...]
    # freqs = freqs[freq_idx:freq_idx+1]

    num_noise_factors = len(noise_factor)
    parameter = {"noise factor" : noise_factor}
    with open(fig_folder.joinpath("parameter.json"), "w") as f:
        json.dump(parameter, f, indent=4)

    folders = []
    for f in range(num_noise_factors):
        print(f"\n\n\nStarting noise factor {f} of {num_noise_factors}")
        mc_fig_folder = run_single_freq_exp(noise, signal, noisy_sig, cov_samples_noisy_signal[:,f,:,:], cov_samples_noise_only[:,f,:,:], cov_samples_signal_only[:,f,:,:], cov_signal[:,f,:,:], cov_noise[:,f,:,:], freqs, dim, rank, rng, base_fig_folder=fig_folder)
        folders.append(mc_fig_folder)

    return fig_folder




if __name__ == "__main__":
    MESHRIR_FOLDER = pathlib.Path("c:/research/datasets/S32-M441_npy")
    sys.path.append(str(MESHRIR_FOLDER))
    import irutilities

    BASE_FIG_FOLDER = pathlib.Path(__file__).parent.joinpath("figs")
    BASE_FIG_FOLDER.mkdir(exist_ok=True)

    #rng = np.random.default_rng(12345654354)
    base_fdr = run_full_speech_exp()

    #base_fdr = pathlib.Path(__file__).parent.joinpath("figs").joinpath("figs_2024_03_04_22_18_0 deg1 1e-2 noisefactor")
    #base_fdr = pathlib.Path("c:/research/papers/2024_eusipco_manifold_cov_estimation/doc/figs").joinpath("figs_2024_03_05_01_17_0_deg1_1e_3noisefactor")
    plot_parameter_exp(base_fdr)

    #exp(5, 1, rng, fp)
    #base_fdr = run_exp_over_degrees_of_freedom()
    # base_fdr = pathlib.Path(__file__).parent.joinpath("figs").joinpath("figs_2024_02_28_16_06_0")
    # plot_parameter_exp(base_fdr)
    # base_fdr = pathlib.Path(__file__).parent.joinpath("figs").joinpath("figs_2024_02_20_12_05_0 exp 2")
    # plot_parameter_exp(base_fdr)
    # base_fdr = pathlib.Path(__file__).parent.joinpath("figs").joinpath("figs_2024_02_20_12_13_0 exp 3")
    #plot_parameter_exp(base_fdr)