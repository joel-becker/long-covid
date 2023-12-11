import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import squigglepy as sq

# Function to calculate alpha and beta for a Beta distribution
def calculate_beta_params(mean, variance):
    common_factor = mean * (1 - mean) / variance - 1
    alpha = mean * common_factor
    beta = (1 - mean) * common_factor
    return alpha, beta

# Function to calculate Gamma distribution parameters from mean and variance
def calculate_gamma_params(mean, variance):
    alpha = mean**2 / variance
    theta = variance / mean
    return alpha, theta

# Function to set up the model
def setup_model(
        time, prevalence, symptom_idx, n_symptoms, 
        baseline_alpha_hyperprior_mean=1, baseline_alpha_hyperprior_var=10, 
        baseline_beta_hyperprior_mean=30, baseline_beta_hyperprior_var=10,
        decayrate_mean_hyperprior_mean=1, decayrate_mean_hyperprior_var=10, decayrate_prior_var=30,
        non_centered=False
    ):
    # Hyperparameters for baseline
    baseline_alpha_alpha, baseline_alpha_theta = calculate_gamma_params(baseline_alpha_hyperprior_mean, baseline_alpha_hyperprior_var)
    baseline_beta_alpha, baseline_beta_theta = calculate_gamma_params(baseline_beta_hyperprior_mean, baseline_beta_hyperprior_var)

    # Hyperparameters for decay rate
    decayrate_alpha_alpha, decayrate_alpha_theta = calculate_gamma_params(decayrate_mean_hyperprior_mean, decayrate_mean_hyperprior_var)

    with pm.Model() as model:
        if non_centered == True:
            # Non-centered priors for baseline parameters
            baseline_alpha_offset = pm.Normal('baseline_alpha_offset', mu=0, sigma=1, shape=n_symptoms)
            baseline_beta_offset = pm.Normal('baseline_beta_offset', mu=0, sigma=1, shape=n_symptoms)

            # Transform to Gamma distribution
            baseline_alpha = pm.Deterministic('baseline_alpha', baseline_alpha_alpha + baseline_alpha_theta * baseline_alpha_offset)
            baseline_beta = pm.Deterministic('baseline_beta', baseline_beta_alpha + baseline_beta_theta * baseline_beta_offset)

            # Non-centered priors for decay rate
            decay_rate_offset = pm.Normal('decay_rate_offset', mu=0, sigma=1, shape=n_symptoms)

            # Transform to Gamma distribution
            decay_rate_alpha = pm.Deterministic('decay_rate_alpha', decayrate_alpha_alpha + decayrate_alpha_theta * decay_rate_offset)
        else:
            # Priors and hyperpriors for baseline parameters
            baseline_alpha = pm.Gamma('baseline_alpha', alpha=baseline_alpha_alpha, beta=1/baseline_alpha_theta, shape=n_symptoms)
            baseline_beta = pm.Gamma('baseline_beta', alpha=baseline_beta_alpha, beta=1/baseline_beta_theta, shape=n_symptoms)

            # Priors and hyperpriors for decay rate
            decay_rate_alpha = pm.Gamma('decay_rate_alpha', alpha=decayrate_alpha_alpha, beta=1/decayrate_alpha_theta, shape=n_symptoms)
        
        baseline = pm.Beta('baseline', alpha=baseline_alpha, beta=baseline_beta, shape=n_symptoms)
        decay_rate = pm.Gamma('decay_rate', alpha=decay_rate_alpha, beta=1/decayrate_prior_var, shape=n_symptoms)

        # Model for prevalence
        prevalence_est = baseline[symptom_idx] * pm.math.exp(-decay_rate[symptom_idx] * time)

        # Likelihood of observations
        Y_obs = pm.Normal('Y_obs', mu=prevalence_est, sigma=0.01, observed=prevalence)

    return model

# Function to prepare data
def prepare_data(df):
    # Reshape data
    melted_df = df.melt(id_vars=['symptom'], var_name='time', value_name='prevalence')
    melted_df['time'] = melted_df['time'].replace({'prevalence_diff_6m': 6, 'prevalence_diff_12m': 12, 'prevalence_diff_18m': 18})
    melted_df['prevalence'] /= 100  # Convert to proportion
    symptom_idx = pd.Categorical(melted_df['symptom']).codes
    return melted_df['time'].values, melted_df['prevalence'].values, symptom_idx

def sample_model(model, draws=1000, tune=500, chains=4, target_accept=0.99):
    """
    Sample from a PyMC model.

    Parameters:
    model (pm.Model): PyMC model to sample from.
    draws (int): The number of samples to draw. Defaults to 1000.
    tune (int): Number of tuning steps. Defaults to 500.
    chains (int): The number of chains to sample. Defaults to 4.
    target_accept (float): Target acceptance probability. Defaults to 0.99.

    Returns:
    pm.MultiTrace: A `MultiTrace` object that contains the samples.
    """

    with model:
        trace = pm.sample(draws, tune=tune, chains=chains, target_accept=target_accept)

    return trace


# Function to plot decay function with uncertainty for all symptoms
def plot_all_symptoms(trace, symptoms, time_points):
    n_symptoms = len(symptoms)
    n_cols = 3
    n_rows = int(np.ceil(n_symptoms / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten to simplify indexing

    for i, symptom in enumerate(symptoms):
        baseline_samples = trace.posterior['baseline'].values[:, :, i]
        decay_rate_samples = trace.posterior['decay_rate'].values[:, :, i]

        # Generate predictions for each sample in the trace
        prevalence_pred = np.array([[baseline * np.exp(-decay_rate * time) for time in time_points] 
                                    for baseline, decay_rate in zip(baseline_samples.flatten(), decay_rate_samples.flatten())])

        percentiles = np.percentile(prevalence_pred, [2.5, 97.5], axis=0)
        axes[i].fill_between(time_points, percentiles[0], percentiles[1], alpha=0.3)
        axes[i].plot(time_points, np.mean(prevalence_pred, axis=0))
        axes[i].set_title(symptom)
        axes[i].set_xlabel('Time (months)')
        axes[i].set_ylabel('Prevalence')

    for ax in axes[n_symptoms:]:  # Hide any unused subplots
        ax.set_visible(False)

    plt.suptitle('Decay of Symptoms Prevalence Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# Example usage
df = adjusted_prevalence
time, prevalence, symptom_idx = prepare_data(df)
model = setup_model(time, prevalence, symptom_idx, n_symptoms=len(df['symptom'].unique()))

# Posterior predictive check
trace = sample_model(model)
symptom_names = df['symptom'].unique().tolist()
plot_all_symptoms(trace, symptom_names, time_points=np.linspace(0, 18, 100))







# Prior predictive check
def plot_implied_paths(model, time_points, samples=3000, plot_bands=True):
    """
    Plots the implied paths over time for the baseline and decay rate parameters with an option
    to plot uncertainty bands.

    Parameters:
    model (pm.Model): The PyMC model.
    time_points (np.array): Array of time points to evaluate the decay function.
    samples (int): Number of samples to draw from the priors.
    plot_bands (bool): Whether to plot uncertainty bands or individual paths.

    Returns:
    None, shows the plots directly.
    """

    with model:
        # Sample from the prior predictive distribution
        prior_predictive = pm.sample_prior_predictive(samples=samples).prior

    # Extract baseline and decay rate samples
    baseline_samples = prior_predictive['baseline'].values.flatten()
    decay_rate_samples = prior_predictive['decay_rate'].values.flatten()

    # Calculate prevalence over time using the decay function for each sample
    prevalence_paths = np.array([baseline * np.exp(-decay_rate * time_points) for baseline, decay_rate in zip(baseline_samples, decay_rate_samples)])

    # Plotting
    plt.figure(figsize=(10, 6))
    if plot_bands:
        # Calculate and plot uncertainty bands
        percentiles = [50, 80, 95]
        for p in percentiles:
            lower = np.percentile(prevalence_paths, 50 - p/2, axis=0)
            upper = np.percentile(prevalence_paths, 50 + p/2, axis=0)
            plt.fill_between(time_points, lower, upper, alpha=0.2, label=f'{p}% Band')
    else:
        # Plot each path with low opacity
        for path in prevalence_paths:
            plt.plot(time_points, path, color='blue', alpha=0.1)

        plt.legend(['Individual Paths'], loc='upper right')

    # Calculate and plot the mean path
    mean_path = np.mean(prevalence_paths, axis=0)
    plt.plot(time_points, mean_path, color='red', label='Mean Path')

    plt.title('Implied Prevalence Paths Over Time')
    plt.xlabel('Time (Months)')
    plt.ylabel('Prevalence')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # Format y-axis as percentage
    plt.xticks(np.arange(0, max(time_points) + 1, 3)) # Discretize x-axis
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Tufte-style grid
    plt.tight_layout()
    plt.show()

# Prior predictive check
time_points = np.linspace(0, 18, 300)  # Adjust as needed (0 to 18 months in this example)
plot_implied_paths(model, time_points)




import numpy as np
import pymc as pm
from scipy.integrate import quad


def calculate_time_adjusted_burden_with_uncertainty(model, severity_proportions, total_cases, symptom_names, num_samples=3000, max_time=36):
    """
    Calculate the severity-adjusted burden over time using the decay model, including uncertainty.

    Parameters:
    model (pm.Model): The PyMC model with decay function parameters.
    severity_proportions (dict): Proportions for severity levels (mild, moderate, severe).
    total_cases (int): Total number of cases.
    symptom_names (list): List of symptom names.
    num_samples (int): Number of samples to draw from the posterior.
    max_time (int): Maximum time to consider (in months).

    Returns:
    dict: Dictionary with symptom names as keys and distributions of the severity-adjusted burden as values.
    """
    burdens = {symptom: [] for symptom in symptom_names}
    burdens['total'] = []

    # Sample from the posterior
    with model:
        trace = pm.sample(num_samples, target_accept=0.99, return_inferencedata=False)

    # Define the decay function for integration
    def decay_func(t, baseline, decay_rate):
        return baseline * np.exp(-decay_rate * t)

    n_symptoms = len(symptom_names)

    # Calculate burden for each sampled pair of baseline and decay rate
    for i in range(num_samples):
        total_burden = 0
        for j in range(n_symptoms):
            baseline = trace['baseline'][i, j]
            decay_rate = trace['decay_rate'][i, j]

            integral, _ = quad(decay_func, 0, max_time, args=(baseline, decay_rate))
            burden = integral * total_cases[i] / 12  # Adjust for total cases and proportion of year
            
            # Adjust for severity
            severity_adjusted_burden = sum(burden * severity_proportions[severity] for severity in severity_proportions)
            
            # Store the burden
            burdens[symptom_names[j]].append(severity_adjusted_burden)
            total_burden += severity_adjusted_burden

        burdens['total'].append(total_burden)

    return burdens


# Plot setup

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_burden_distributions(burdens):
    """
    Plots the distributions of burdens.

    Parameters:
    burdens (dict): Dictionary with symptom names as keys and distributions of burdens as values.
    """

    # Plot setup
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Plot for each symptom
    for symptom, values in burdens.items():
        if symptom != 'total':
            axes[0].hist(values, alpha=0.5, label=symptom, bins=30)
    
    axes[0].set_title('Burden Distributions by Symptom')
    axes[0].set_xlabel('Symptom Burden')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Plot for total burden
    axes[1].hist(burdens['total'], bins=30, color='green')  # Increased number of bins
    axes[1].set_title('Burden Across Symptoms (DALYs)')
    axes[1].set_xlabel('Total Burden (DALYs)')

    # Format x-axis labels and rotate
    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)  # Remove y-axis elements

    plt.tight_layout()
    plt.show()

num_samples=3000
annual_cases = sq.to(7000000, 40000000, credibility=99) @ num_samples
symptom_names = list(adjusted_prevalence['symptom'].unique())
burdens = calculate_time_adjusted_burden_with_uncertainty(model, severity_proportions, annual_cases, symptom_names)
plot_burden_distributions(burdens)
