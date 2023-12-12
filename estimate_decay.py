import pymc as pm
import pandas as pd

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
        baseline_alpha_hyperprior_mean=1, baseline_alpha_hyperprior_var=3, 
        baseline_beta_hyperprior_mean=50, baseline_beta_hyperprior_var=10,
        decayrate_mean_hyperprior_mean=1, decayrate_mean_hyperprior_var=10, decayrate_prior_var=10,
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
    melted_df['prevalence'] = pd.to_numeric(melted_df['prevalence'], errors='coerce')
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
