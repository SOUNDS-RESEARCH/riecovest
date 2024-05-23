
import numpy as np
import pathlib
import scipy.linalg as splin
import scipy.stats as spstat

import json

import matplotlib.pyplot as plt
import aspsim.diagnostics.plot as aspplot
import aspsim.fileutilities as futil
import aspsim.diagnostics.plot as dplot

import aspcol.distance as aspdist

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
    
    if base_fig_folder is None:
        base_fig_folder = pathlib.Path(__file__).parent.joinpath("figs")
        base_fig_folder.mkdir(exist_ok=True)

    fig_folder = futil.get_unique_folder_name("figs_", base_fig_folder)
    fig_folder.mkdir()
    
    num_cov_data = 64
    num_data = 1000
    snr_db = 0
    snr_lin = 10**(snr_db / 10)
    condition = 1e-2

    noisy_sig, signal, noise, cov_samples_noisy_signal, cov_samples_noise_only, cov_signal, cov_noise = gen_sampled_data(dim, rank, snr_lin, num_cov_data, num_data, complex_data, noise_dist, signal_dist, rng, degrees_of_freedom=degrees_of_freedom, condition=condition)

    sim_info = {}
    sim_info[f"Noise power"] = np.mean(np.abs(noise)**2)
    sim_info[f"Signal power"] = np.mean(np.abs(signal)**2)
    sim_info[f"Prior SNR (dB)"] = 10 * np.log10(np.mean(np.abs(signal)**2) / np.mean(np.abs(noise)**2))
    sim_info[f"Data points for cov estimation"] = num_cov_data
    sim_info[f"Dimension"] = dim
    sim_info[f"Signal rank"] = rank
    sim_info[f"Noise distribution"] = noise_dist
    sim_info[f"Signal distribution"] = signal_dist
    sim_info[f"Degrees of freedom"] = degrees_of_freedom
    sim_info[f"complex data"] = complex_data
    sim_info[f"condition number"] = condition
    with open(fig_folder.joinpath("sim_info.json"),"w") as f:
        json.dump(sim_info, f, indent=4)

    # ========= SOLVE ESIMTATION PROBLEM BELOW HERE ============
    est_cov_sig, est_cov_noise = estimate_all(noisy_sig, signal, noise, cov_samples_noisy_signal, cov_samples_noise_only, rank, cov_signal, cov_noise)
    est_cov_sig, est_cov_noise, cov_signal, cov_noise = normalize_cov_matrices(est_cov_sig, est_cov_noise, cov_signal, cov_noise)
    
    all_cov_sig = {**est_cov_sig}
    all_cov_sig["true"] = cov_signal
    all_cov_noise = {**est_cov_noise}
    all_cov_noise["true"] = cov_noise

    np.savez(fig_folder.joinpath("cov_sig"),  **all_cov_sig)
    np.savez(fig_folder.joinpath("cov_noise"),  **all_cov_noise)
    
    #estimation_errors(est_cov_sig, cov_signal, fig_folder, "signal_covariance")
    #estimation_errors(est_cov_noise, cov_noise, fig_folder, "noise_covariance")
    estimation_errors_all(est_cov_sig, est_cov_noise, cov_signal, cov_noise, fig_folder)
    
    show_matrices(all_cov_sig, fig_folder, "signal_covariance")
    show_matrices(all_cov_noise, fig_folder, "noise_covariance")
    #show_rank(est_cov_sig, fig_folder, "signal_covariance")
    show_eigenvalues(all_cov_sig, fig_folder, "signal")
    show_eigenvalues(all_cov_noise, fig_folder, "noise")



def estimate_all(noisy_sig, signal, noise, cov_samples_noisy_signal, cov_samples_noise_only, true_sig_rank, true_signal_cov, true_noise_cov):
    scm_noisy_sig = covest.scm(cov_samples_noisy_signal)
    scm_noise = covest.scm(cov_samples_noise_only)
    tem_noisy_sig = covest.tyler_estimator(cov_samples_noisy_signal, 20)
    tem_noisy_sig = tem_noisy_sig / np.trace(tem_noisy_sig) * np.trace(scm_noisy_sig)
    tem_noise = covest.tyler_estimator(cov_samples_noise_only, 20)
    tem_noise = tem_noisy_sig / np.trace(tem_noise) * np.trace(scm_noise)

    estimator = {}
    cov_sig = {}
    cov_noise = {}
    #cov["wasserstein"], cov_noise["wasserstein"] = est_manifold_wasserstein(scm_noisy_sig, scm_noise, true_sig_rank)
    
    cov_sig["subtract-scm"] = scm_noisy_sig - scm_noise
    cov_sig["gevd"] = covest.est_gevd(scm_noisy_sig, scm_noise, true_sig_rank)
    cov_sig["gevd tyler"] = covest.est_gevd(tem_noisy_sig, tem_noise, true_sig_rank)
    #cov_sig["tyler-scale"], cov_noise["tyler-scale"] = covest.est_manifold_tylers_with_scale_opt(cov_samples_noisy_signal, cov_samples_noise_only, true_sig_rank)
    cov_sig["tyler"], cov_noise["tyler"] = covest.est_manifold_tylers_m_estimator(cov_samples_noisy_signal, cov_samples_noise_only, true_sig_rank)#, start_value = (cov_sig["gevd"], scm_noise))
    #cov_sig["tyler nonorm"], cov_noise["tyler nonorm"] = covest.est_manifold_tylers_m_estimator_nonorm(cov_samples_noisy_signal, cov_samples_noise_only, true_sig_rank)#, start_value = (cov_sig["gevd"], scm_noise))
    #cov_sig["wishart"], cov_noise["wishart"] = covest.est_manifold_wishart(scm_noisy_sig, scm_noise, true_sig_rank, 12)#, start_value = (cov_sig["gevd"], scm_noise))
    #cov_sig["airm"], cov_noise["airm"] = covest.est_manifold_airm(scm_noisy_sig, scm_noise, true_sig_rank)#, start_value = (cov_sig["gevd"], scm_noise))
    
    #cov_sig["frob"], cov_noise["frob"] = covest.est_manifold_frob(scm_noisy_sig, scm_noise, true_sig_rank)
    #cov_sig["frob_whitened"], cov_noise["frob_whitened"] = est_manifold_frob_whitened(scm_noisy_sig, scm_noise, true_sig_rank)
    #cov_sig["frob_whitened_opt"], cov_noise["frob_whitened_opt"] = est_manifold_frob_whitened_by_true_noise(scm_noisy_sig, scm_noise, true_sig_rank)
    
    #cov_sig["wishart"] *= np.trace(cov_sig["frob"]) / np.trace(cov_sig["wishart"])
    #cov_noise["wishart"] *= np.trace(cov_noise["frob"]) / np.trace(cov_noise["wishart"])

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
        
    
    dplot.output_plot("pdf", fig_folder, f"matrices_{name}")

def show_eigenvalues(mat_dict, fig_folder, name = ""):

    fig, ax = plt.subplots(1,1, figsize=(8, 3))
    for i, (est_name, mat) in enumerate(mat_dict.items()):
        eigvals = np.linalg.eigvalsh(mat)
        ax.plot(eigvals, "-o", label=est_name)
        ax.set_xlabel("Eigenvalue index")
        ax.legend()

        dplot.set_basic_plot_look(ax)
    dplot.output_plot("pdf", fig_folder, f"eigenvalues_{name}")
    


def estimation_errors_all(cov_sig, cov_noise, true_signal_cov, true_noise_cov, fig_folder):
    true_noisy_sig_cov = true_signal_cov + true_noise_cov

    estimation_errors(cov_sig, true_signal_cov, False, fig_folder, "signal_covariance")
    estimation_errors(cov_noise, true_noise_cov, True, fig_folder, "noise_covariance")

    cov_noisy_sig = {}
    for name, Rx in cov_sig.items():
        if name in cov_noise:
            Rv = cov_noise[name]
        else:
            Rv = cov_noise["scm"]
        cov_noisy_sig[name] = Rx + Rv
        #Ry_est = Rx + Rv
    estimation_errors(cov_noisy_sig, true_noisy_sig_cov, True, fig_folder, "noisy_signal_covariance")




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
        
        dim = cov_true.shape[-1]
        cov_true_trace_normalized = cov_true * dim / np.trace(cov_true)
        #cov_est_trace_normalized = cov_est * (np.trace(cov_true) / np.trace(cov_est))
        cov_est_trace_normalized = cov_est * dim / np.trace(cov_est)
        metrics["mse trace-normalized"][est_name] = np.linalg.norm(cov_est_trace_normalized - cov_true_trace_normalized, ord="fro")**2
        metrics["nmse trace-normalized"][est_name] = np.linalg.norm(cov_est_trace_normalized - cov_true_trace_normalized, ord="fro")**2 / np.linalg.norm(cov_true_trace_normalized, ord="fro")**2
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





def run_monte_carlo_trial(num_trials, dim, rank, complex_data, noise_dist, signal_dist, degrees_of_freedom=None, base_fig_folder=None):
    if base_fig_folder is None:
        base_fig_folder = pathlib.Path(__file__).parent.joinpath("figs")
        base_fig_folder.mkdir(exist_ok=True)
    fig_folder = futil.get_unique_folder_name("figs_mc_", base_fig_folder)
    fig_folder.mkdir()

    rng = np.random.default_rng(123456789)
    for i in range(num_trials):
        exp(dim, rank, complex_data, noise_dist, signal_dist, rng=rng, base_fig_folder=fig_folder, degrees_of_freedom=degrees_of_freedom)

    return fig_folder







def analyze_monte_carlo_trial(base_fig_folder):
    cov_sig_all = []
    cov_noise_all = []
    sim_info_all = []
    est_errors_all = []
    est_errors_noise_covariance = []
    est_errors_noisy_signal_covariance = []

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





def run_exp_over_degrees_of_freedom():
    base_fig_folder = pathlib.Path(__file__).parent.joinpath("figs")
    base_fig_folder.mkdir(exist_ok=True)

    fig_folder = futil.get_unique_folder_name("figs_", base_fig_folder)
    fig_folder.mkdir()

    dim = 10
    rank = 3
    num_samples = 20
    dof = [1, 2, 3, 4, 5, 10, 20, 50]


    parameter = {"dof" : dof}
    with open(fig_folder.joinpath("parameter.json"), "w") as f:
        json.dump(parameter, f, indent=4)

    folders = []
    for dof_val in dof:
        mc_fig_folder = run_monte_carlo_trial(num_samples, dim, rank, complex_data=True, noise_dist="t-distribution", signal_dist="gaussian", degrees_of_freedom=dof_val, base_fig_folder=fig_folder)
        folders.append(mc_fig_folder)

    return fig_folder


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
        for nm in algo_names:
            ax.plot(parameter_vals, sum_dict[f"{nm} : sample mean"], label={nm})
        
        y_lim = ax.get_ylim()
        for i, nm in enumerate(algo_names):
            center = np.array(sum_dict[f"{nm} : sample mean"])
            width = np.sqrt(np.array(sum_dict[f"{nm} : variance of sample mean"]))
            ax.fill_between(np.array(parameter_vals), center-width, center+width, alpha=0.4, color=f"C{i}")
        ax.set_ylim(y_lim)

        ax.legend()
        ax.set_xlabel(parameter_name)
        ax.set_ylabel(sum_name)
        aspplot.set_basic_plot_look(ax)
        aspplot.output_plot("tikz", fig_folder, f"{sum_name}_{parameter_name}")

        fig, ax = plt.subplots(1,1, figsize=(8,6))
        for nm in algo_names:
            ax.plot(parameter_vals, 10 * np.log10(sum_dict[f"{nm} : sample mean"]), label={nm})

        y_lim = ax.get_ylim()
        for i, nm in enumerate(algo_names):
            center = np.array(sum_dict[f"{nm} : sample mean"])
            width =  np.sqrt(np.array(sum_dict[f"{nm} : variance of sample mean"]))
            lower_lim = center-width
            lower_lim[lower_lim <= 0] = 1e-10
            upper_lim = center + width
            ax.fill_between(np.array(parameter_vals), 10*np.log10(lower_lim), 10*np.log10(upper_lim), alpha=0.4, color=f"C{i}")
        ax.set_ylim(y_lim)

        ax.legend()
        ax.set_xlabel(f"{parameter_name}")
        ax.set_ylabel(f"{sum_name} (dB)")
        aspplot.set_basic_plot_look(ax)
        aspplot.output_plot("tikz", fig_folder, f"{sum_name}_{parameter_name}_db")



def plot_snr_for_exp(fig_folder):
    mc_folders = []
    for fdr in fig_folder.iterdir():
        if fdr.is_dir():
            if fdr.stem.startswith("figs_mc_"):
                mc_folders.append(fdr)

    with open(fig_folder.joinpath("parameter.json")) as f:
        parameter = json.load(f)
        parameter_name = list(parameter.keys())[0]
        parameter_vals = parameter[parameter_name]

    noise_powers = []
    signal_powers = []
    for mc_fdr in mc_folders:
        for fdr in mc_fdr.iterdir():
            if fdr.is_dir():
                noise_pwr = []
                sig_pwr = []
                with open(fdr.joinpath("sim_info.json")) as f:
                    sim_info = json.load(f)
                    noise_pwr.append(sim_info["Noise power"])
                    sig_pwr.append(sim_info["Signal power"])
        noise_powers.append(np.mean(noise_pwr))
        signal_powers.append(np.mean(sig_pwr))
    noise_powers = np.array(noise_powers)
    signal_powers = np.array(signal_powers)
    snr = signal_powers / noise_powers

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(parameter_vals, snr)
    
    # y_lim = ax.get_ylim()
    # for i, nm in enumerate(algo_names):
    #     center = np.array(sum_dict[f"{nm} : sample mean"])
    #     width = np.sqrt(np.array(sum_dict[f"{nm} : variance of sample mean"]))
    #     ax.fill_between(np.array(parameter_vals), center-width, center+width, alpha=0.4, color=f"C{i}")
    # ax.set_ylim(y_lim)
    #ax.legend()
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Signal to noise ratio")
    aspplot.set_basic_plot_look(ax)
    aspplot.output_plot("tikz", fig_folder, f"snr_{parameter_name}")

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(parameter_vals, 10*np.log10(snr))
    
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Signal to noise ratio")
    aspplot.set_basic_plot_look(ax)
    aspplot.output_plot("tikz", fig_folder, f"snr_{parameter_name}_db")






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


if __name__ == "__main__":
    #base_fdr = run_exp_over_degrees_of_freedom()
    #base_fdr = pathlib.Path(__file__).parent.joinpath("figs").joinpath("figs_2024_02_28_00_33_0 condition 1e-1 n12")
    # plot_parameter_exp(base_fdr)
    base_fdr = pathlib.Path(__file__).parent.joinpath("figs").joinpath("figs_2024_02_27_10_22_0")
    # plot_parameter_exp(base_fdr)
    #base_fdr = pathlib.Path("C:/research/papers/2024_eusipco_manifold_cov_estimation/doc/figs/2024_02_21_experiment/figs_2024_02_21_00_16_0_exp4")
    plot_parameter_exp(base_fdr)
    plot_snr_for_exp(base_fdr)
