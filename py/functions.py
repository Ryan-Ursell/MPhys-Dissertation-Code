import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.integrate as integrate
from collections import defaultdict
from scipy.stats import gaussian_kde
from bilby.core.result import read_in_result
from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
from pesummary.gw.conversions.spins import viewing_angle_from_inclination


# Data loading functions
def data_loader(loc, label, get_viewing_angle=True):
    data = MultiAnalysisSamplesDict.from_files(loc, disable_prior=True)
    injection_values = read_in_result(loc[label[0]]).injection_parameters
    if get_viewing_angle:
        data[label[0]]['viewing_angle'] = viewing_angle_from_inclination(data[label[0]]['theta_jn'])
        data[label[1]]['viewing_angle'] = viewing_angle_from_inclination(data[label[1]]['theta_jn'])
        injection_values['viewing_angle'] = viewing_angle_from_inclination(injection_values['theta_jn'])
    return data, injection_values

def generate_filename(label, parent_file, param):
    return {
        label[0]: f"{parent_file}/final_result/{label[0]}_{param}_data0_1126259642-413_analysis_H1L1V1_merge_result.json",
        label[1]: f"{parent_file}/final_result/{label[1]}_{param}_data0_1126259642-413_analysis_H1L1V1_merge_result.json"
    }

def load_data(parameter_list, label, parent_file, get_viewing_angle=True):
    data_dict = {}
    for param in parameter_list:
        loc = generate_filename(label, parent_file, param)
        data_dict[param] = data_loader(loc, label, get_viewing_angle=get_viewing_angle)

    return data_dict


# General plotting functions
## Plot 2D posterior distributions
def plot_2d(data, injected_values, injection_labels, param_list, title,
            contour_colours=['tab:blue', 'tab:orange'], custom_labels=None, save_path=None):

    if all(param in injection_labels for param in param_list):
        injected_x = injected_values[param_list[0]]
        injected_y = injected_values[param_list[1]]

        fig, ax1, ax2, ax3 = data.plot(parameters=param_list, type="reverse_triangle", levels=[0.95], colors=contour_colours, plot_datapoints=False)
        ax2.set_title(title)

        line1 = ax2.axvline(injected_x, color="black", linestyle="-")
        ax3.axvline(injected_x, color="black", linestyle="-")
        line3 = ax1.axhline(injected_y, color="black", linestyle="--")
        ax2.axhline(injected_y, color="black", linestyle="--")

        legend_elements = [Line2D([0], [0], color=contour_colours[0], lw=2, label='Unbiased'), 
                           Line2D([0], [0], color=contour_colours[1], lw=2, label='Biased'), 
                           Line2D([0], [0], color="black", linestyle="-", label=f"Injected {param_list[0]}"), 
                           Line2D([0], [0], color="black", linestyle="--", dashes=(1, 1), label=f"Injected {param_list[1]}")]
        ax2.legend(handles=legend_elements, loc="best", handlelength=2.5)
        

    else:
        data.plot(parameters=param_list, type="reverse_triangle")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", transparent=True)

    plt.show()

## Plot differences in bias between runs
def extract_diff_by_run(results, run_label):
    """
    Extracts differences between biased and unbiased probabilities for each parameter in a single run.
    """
    param_probs = defaultdict(dict)
    pattern = re.compile(r"(?P<param>\w+)_probability_(?P<bias>biased|unbiased) for (?P<sigma>[\d.]+ sigma)")

    for key, value in results.items():
        match = pattern.match(key)
        if match:
            param = match.group("param")
            bias = match.group("bias")
            sigma = match.group("sigma")
            full_param_key = f"{param} for {sigma}"
            param_probs[full_param_key][bias] = value
            
    diff_dict = {}
    for param, biases in param_probs.items():
        if 'biased' in biases and 'unbiased' in biases:
            diff = biases['unbiased'] - biases['biased']
            diff_dict[param] = diff/biases['unbiased']

    return {run_label: diff_dict}

def compare_multiple_runs(all_results):
    """
    Takes a dictionary of run_name -> results_dict
    Returns: run_name -> param -> diff
    """
    full_diff_dict = {}
    for run_label, result_dict in all_results.items():
        run_diff = extract_diff_by_run(result_dict, run_label)
        full_diff_dict.update(run_diff)

    return full_diff_dict

def format_param(param):
    param = param.replace("probability ", "").split(" for")[0]
    parts = param.split("_")

    if len(parts) > 2:
        formatted_name = " vs ".join(["_".join(parts[i:i+2]) for i in range(0, len(parts), 2)])
    else:
        formatted_name = "_".join(parts)
    
    return formatted_name

def plot_bias_differences_by_parameter(diff_dict_by_run, title_name):
    all_params = sorted({param for run in diff_dict_by_run.values() for param in run})

    cleaned_labels = [format_param(param) for param in all_params]
    
    runs = list(diff_dict_by_run.keys())
    num_runs = len(runs)
    num_params = len(all_params)
    
    bar_height = 0.8 / num_runs
    y_positions = np.arange(num_params)
    
    fig, ax = plt.subplots(figsize=(12, max(6, num_params * 0.3)))
    colors = plt.cm.tab10.colors

    for i, run in enumerate(runs):
        diffs = [diff_dict_by_run[run].get(param, 0) for param in all_params]
        offset = (i - num_runs / 2) * bar_height + bar_height / 2
        ax.barh(y_positions + offset, diffs, height=bar_height, label=run, color=colors[i % len(colors)])

    ax.set_yticks(y_positions)
    ax.set_yticklabels(cleaned_labels)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Unbiased - Biased Probability Difference")
    ax.set_title(f"Difference in Probabilities (Unbiased - Biased) per Parameter for {title_name}")
    ax.legend(title="Run")

    plt.tight_layout()
    plt.show()

def plot_nd_heatmap(diff_dict_by_run, title_name, save_path=None):
    df = pd.DataFrame(diff_dict_by_run).fillna(0)

    df.index = [format_param(param) for param in df.index]

    plt.figure(figsize=(10, len(df)))
    sns.heatmap(df, cmap="coolwarm_r", center=0, annot=True, fmt=".2f", linewidths=0.5)
    plt.title(f"Unbiased - Biased Differences (N-Dimensional Heatmap) for {title_name}")
    plt.xlabel("Run")
    plt.ylabel("Parameter Pair")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf")

    plt.show()

# Bias quantification functions
## Shifted distribution method
### Compute the probabilities
def generate_random_points_below_kde(data_nd, n_samples):
    kde = gaussian_kde(data_nd.T)
    return kde.resample(n_samples).T

def calculate_nd_probability(param_names, data_nd, injected_point, n_samples, sigma_multiplier, show_plots=False, bias_label=''):
    accepted_samples = generate_random_points_below_kde(data_nd, n_samples)
    original_mean = np.mean(data_nd, axis=0)
    cov_matrix = np.cov(data_nd, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix)
    
    count_within_sigma = 0
    for shift_point in accepted_samples:
        shift_vector = shift_point - original_mean
        shifted_mean = original_mean + shift_vector

        diff = injected_point - shifted_mean
        mahal_dist = np.sqrt(diff.T @ cov_inv @ diff)

        if mahal_dist <= sigma_multiplier:
            count_within_sigma += 1

        if show_plots and data_nd.shape[1] == 2:
            shifted_data = data_nd + shift_vector
            plt.scatter(shifted_data[:, 0], shifted_data[:, 1], alpha=0.02, s=1)

    if show_plots and data_nd.shape[1] == 2:
        plt.scatter(*injected_point, color='red', label='Injected Value', zorder=5)
        plt.title(f'{bias_label} {param_names[0]} vs {param_names[1]}')
        plt.xlabel(param_names[0])
        plt.ylabel(param_names[1])
        plt.legend()
        plt.savefig('./nd_contour', format="pdf")
        plt.show()

    return count_within_sigma / n_samples

def compute_nd_probabilities(data, injected_values, param_groups, num_loops=10000, sigma_multiplier=1.6, show_plots=False):
    results = {}

    for params in param_groups:
        if all(param in injected_values for param in params):
            injected_point = np.array([injected_values[param] for param in params])

            if all(param in data['unbiased'] for param in params):
                data_unbiased = np.array([data['unbiased'][param] for param in params]).T
                prob_unbiased = calculate_nd_probability(
                    params, data_unbiased, injected_point,
                    num_loops, sigma_multiplier, show_plots, bias_label='unbiased'
                )
                results[f"{'_'.join(params)}_probability_unbiased for {sigma_multiplier} sigma"] = prob_unbiased

            if all(param in data['biased'] for param in params):
                data_biased = np.array([data['biased'][param] for param in params]).T
                prob_biased = calculate_nd_probability(
                    params, data_biased, injected_point,
                    num_loops, sigma_multiplier, show_plots, bias_label='biased'
                )
                results[f"{'_'.join(params)}_probability_biased for {sigma_multiplier} sigma"] = prob_biased

    return results

### Plot shifted distribution probabiliies
def plot_nd_comparison(results_dict, title_name, save_path=None):
    param_labels, unbiased_probs, biased_probs = [], [], []

    for key, value in results_dict.items():
        match = re.match(r"(.+?)_probability_(unbiased|biased)", key)
        if not match:
            continue

        param_key, bias_type = match.groups()
        parts = param_key.split("_")
        
        param_names = ["_".join(parts[i:i+2]) for i in range(0, len(parts), 2)]
        formatted_name = " vs ".join(param_names)

        if bias_type == "unbiased":
            param_labels.append(formatted_name)
            unbiased_probs.append(value)
        elif bias_type == "biased":
            biased_probs.append(value)

    x = np.arange(len(param_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, max(6, len(param_labels) * 0.4)))
    ax.barh(x - width / 2, unbiased_probs, width, label='Unbiased', color='skyblue')
    ax.barh(x + width / 2, biased_probs, width, label='Biased', color='lightcoral')

    ax.set_xlabel('Probability')
    ax.set_title(f'Comparison of Unbiased vs Biased Probabilities for {title_name}')
    ax.set_yticks(x)
    ax.set_yticklabels(param_labels)
    ax.legend()

    if save_path:
        fig.savefig(save_path, format="pdf", transparent=True)

    plt.tight_layout()
    plt.show()

## Cost function method
### Compute the costs
def cost_function(theta_values, theta_0):
    kde = gaussian_kde(theta_values)
    lower = np.min(theta_values)
    upper = np.max(theta_values)
    
    def integrand(theta):
        return kde(theta) * (theta - theta_0)**2

    cost, error = integrate.quad(integrand, lower, upper)
    return np.sqrt(cost)

def compute_costs(data, injected_values):
    results, results_norm = {}, {}

    for param in injected_values:
        if param == 'reference_frequency' or param == 'minimum_frequency':
            continue
        elif param in data['unbiased'] and param in data['biased']:
            unbiased_theta = data['unbiased'][param]
            biased_theta = data['biased'][param]
            theta_0 = injected_values[param]

            std_unbiased = np.std(unbiased_theta)
            std_biased = np.std(biased_theta)

            results[f"{param}_unbiased"] = {
                "cost": cost_function(unbiased_theta, theta_0),
                "norm_cost": std_unbiased / cost_function(unbiased_theta, theta_0)
            }

            results[f"{param}_biased"] = {
                "cost": cost_function(biased_theta, theta_0),
                "norm_cost": std_biased / cost_function(biased_theta, theta_0)
            }

    return results

## Plotting costs
def print_costs(results, name):
    print(f"{name}:")
    for param, values in results.items():
        print(f"{param}_cost: {values['cost']}")
        print(f"{param}_norm_cost: {values['norm_cost']}")
        print()
    print()

def plot_costs(results_dict, title="Cost Function Comparison", show_cost=False, show_norm_cost=True):
    params = sorted(set('_'.join(k.split('_')[:-1]) for k in results_dict.keys()))
    y = range(len(params))

    num_plots = int(show_cost) + int(show_norm_cost)
    fig, axs = plt.subplots(1, num_plots, figsize=(10 * num_plots, len(params) * 0.5 + 2), sharey=True)
    if num_plots == 1:
        axs = [axs]

    plot_idx = 0

    if show_cost:
        cost_unbiased = [results_dict[f"{p}_unbiased"]['cost'] for p in params]
        cost_biased = [results_dict[f"{p}_biased"]['cost'] for p in params]

        axs[plot_idx].barh(y=[i - 0.2 for i in y], width=cost_unbiased, height=0.4, label='Unbiased', color='skyblue')
        axs[plot_idx].barh(y=[i + 0.2 for i in y], width=cost_biased, height=0.4, label='Biased', color='salmon')
        axs[plot_idx].set_yticks(y)
        axs[plot_idx].set_yticklabels(params)
        axs[plot_idx].tick_params(labelleft=True)
        axs[plot_idx].set_xlabel("Cost")
        axs[plot_idx].set_title(f"{title} - Raw Cost")
        axs[plot_idx].legend()
        axs[plot_idx].grid(True, linestyle='--', alpha=0.4)
        plot_idx += 1

    if show_norm_cost:
        norm_cost_unbiased = [results_dict[f"{p}_unbiased"]['norm_cost'] for p in params]
        norm_cost_biased = [results_dict[f"{p}_biased"]['norm_cost'] for p in params]

        axs[plot_idx].barh(y=[i - 0.2 for i in y], width=norm_cost_unbiased, height=0.4, label='Unbiased', color='skyblue')
        axs[plot_idx].barh(y=[i + 0.2 for i in y], width=norm_cost_biased, height=0.4, label='Biased', color='salmon')
        axs[plot_idx].set_yticks(y)
        axs[plot_idx].set_yticklabels(params)
        #axs[plot_idx].tick_params(labelleft=True)
        axs[plot_idx].set_xlabel("Normalized Cost (σ / cost)")
        axs[plot_idx].set_title(f"{title} - Normalized Cost")
        axs[plot_idx].legend()
        axs[plot_idx].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()


## Bayes factor method
### Compute the Bayes factor
def bayes_factor(loc):
    result_unbiased = read_in_result(loc['unbiased'])
    result_biased = read_in_result(loc['biased'])
    logz_unbiased = result_unbiased.log_evidence
    logz_biased = result_biased.log_evidence

    log_bf = logz_unbiased - logz_biased
    bf = np.exp(log_bf)

    return log_bf, bf

### Plot the Bayes factor
def print_bayes_factor(log_bf, bf, name):
    print(f"{name}:")
    print(f"Log Bayes Factor (unbiased vs biased): {log_bf:.2f}")
    print(f"Bayes Factor: {bf:.2f}")
    print()

def bf_plotter(bf, log_bf, run_names, show_bf=True, show_log_bf=True, title="Bayes Factor Comparison"):
    plots_to_show = sum([show_bf, show_log_bf])
    fig, axs = plt.subplots(1, plots_to_show, figsize=(7 * plots_to_show, 5))
    ax_idx = 0
    
    if plots_to_show == 1:
        axs = [axs]
    
    if show_bf:
        axs[ax_idx].plot(bf, marker='o', linestyle='-', color='blue', label='Bayes Factor')
        axs[ax_idx].set_title('Bayes Factor Comparison')
        axs[ax_idx].set_xlabel("Run")
        axs[ax_idx].set_ylabel("Bayes Factor (10^18)")
        axs[ax_idx].set_xticks(range(len(run_names)))
        axs[ax_idx].set_xticklabels(run_names)
        axs[ax_idx].legend()
        ax_idx += 1

    if show_log_bf:
        axs[ax_idx].plot(log_bf, marker='o', linestyle='-', color='red', label='Log Bayes Factor')
        axs[ax_idx].set_title('Log Bayes Factor Comparison')
        axs[ax_idx].set_xlabel("Run")
        axs[ax_idx].set_ylabel("Log Bayes Factor")
        axs[ax_idx].set_xticks(range(len(run_names)))
        axs[ax_idx].set_xticklabels(run_names)
        axs[ax_idx].legend()

    plt.tight_layout()
    plt.suptitle(title)
    plt.subplots_adjust(top=0.9)
    plt.show()