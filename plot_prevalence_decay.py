import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from scipy.integrate import quad


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
        percentiles = [50, 80, 95, 99]
        for p in percentiles:
            lower = np.percentile(prevalence_paths, 50 - p/2, axis=0)
            upper = np.percentile(prevalence_paths, 50 + p/2, axis=0)
            plt.fill_between(time_points, lower, upper, alpha=0.2, label=f'{p}% Band')
    else:
        # Plot each path with low opacity
        for i, path in enumerate(prevalence_paths):
            if i == 0:
                # Label only the first path
                plt.plot(time_points, path, color='purple', alpha=0.01, label='Individual Paths')
            else:
                # No label for the rest of the paths
                plt.plot(time_points, path, color='purple', alpha=0.01)
        
        plt.legend(loc='upper right')

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


def calculate_time_adjusted_burden_with_uncertainty(model, severity_proportions, total_cases, symptom_daly_data, num_samples=3000, max_time=36):
    """
    Calculate the severity-adjusted burden over time using the decay model, including uncertainty.

    Parameters:
    model (pm.Model): The PyMC model with decay function parameters.
    severity_proportions (dict): Proportions for severity levels (mild, moderate, severe).
    total_cases (int): Total number of cases.
    symptom_daly_data (list): Data on symptoms and their respective DALY adjustments.
    num_samples (int): Number of samples to draw from the posterior.
    max_time (int): Maximum time to consider (in months).

    Returns:
    dict: Dictionary with symptom names as keys and distributions of the severity-adjusted burden as values.
    """
    symptom_names = symptom_daly_data['symptom'].unique().tolist()
    burdens = {symptom: [] for symptom in symptom_names}
    burdens['total'] = []

    # Sample from the posterior
    with model:
        trace = pm.sample(num_samples, target_accept=0.99, return_inferencedata=False)

    # Define the decay function for integration
    def decay_func(t, baseline, decay_rate):
        return baseline * np.exp(-decay_rate * t)

    n_symptoms = len(symptom_names)

    for i in range(num_samples):
        total_burden = 0

        for j in range(n_symptoms):
            symptom_specific_daly_data = symptom_daly_data[symptom_daly_data['symptom'] == symptom_names[j]].copy()
            symptom_specific_daly_data['severity_proportion'] = symptom_specific_daly_data.apply(
                lambda row: severity_proportions['mild'] * row['mild'] +
                            severity_proportions['moderate'] * row['moderate'] +
                            severity_proportions['severe'] * row['severe'],
                axis=1
            )
            symptom_weighted_daly = symptom_specific_daly_data['daly_adjustment'] * symptom_specific_daly_data['severity_proportion']

            baseline = trace['baseline'][i, j]
            decay_rate = trace['decay_rate'][i, j]

            integral, _ = quad(decay_func, 0, max_time, args=(baseline, decay_rate))
            burden = integral * total_cases[i] / 12  # Adjust for total cases and proportion of year
            
            # Adjust for severity
            severity_adjusted_burden = sum(burden * symptom_weighted_daly)
            
            # Store the burden for each symptom
            burdens[symptom_names[j]].append(severity_adjusted_burden)
            total_burden += severity_adjusted_burden

        # Store the total burden for each sample
        burdens['total'].append(total_burden)

    return burdens


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

