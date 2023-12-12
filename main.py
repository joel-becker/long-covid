import prepare as prep
import wrangle_daly_adjustment as wda
import adjust_prevalence as ap
import calculate_burden as cb
import plot_raw_prevalence as plot_rp
import estimate_decay as ed
import plot_prevalence_decay as plot_pd
import numpy as np
import pandas as pd
import squigglepy as sq


# Load data from CSV files
prevalence_diff = prep.prepare_raw_data()
data_daly = pd.read_csv('data/daly.csv')


# Wrangle DALY data
data_daly = wda.wrangle_daly_data(data_daly)


# Adjusting the prevalence data
conservative_prevalence = ap.adjust_prevalence_data(prevalence_diff, method='conservative')
moderate_prevalence = ap.adjust_prevalence_data(prevalence_diff, method='moderate')


# Merging data
conservative_prevalence_merged = cb.merge_prevalence_severity(conservative_prevalence, data_daly)
conservative_prevalence_merged = cb.calculate_period_burden(conservative_prevalence_merged)

moderate_prevalence_merged = cb.merge_prevalence_severity(moderate_prevalence, data_daly)
moderate_prevalence_merged = cb.calculate_period_burden(moderate_prevalence_merged)


# Print total burden in different severity scenarios
severity_proportions_mild = {'mild': 1, 'moderate': 0, 'severe': 0}
severity_proportions_moderate = {'mild': 0, 'moderate': 1, 'severe': 0}
severity_proportions_severe = {'mild': 0, 'moderate': 0, 'severe': 1}
severity_proportions_viv = {'mild': 0.9199, 'moderate': 0.0714, 'severe': 0.0088}
severity_proportions_raddad = {'mild': 0.9975, 'moderate': 0.0023, 'severe': 0.0002}
severity_proportions_robinson = {'mild': 0.94, 'moderate': 0.047, 'severe': 0.013}

severity_scenarios = {
    'Mild': severity_proportions_mild,
    'Moderate': severity_proportions_moderate,
    'Severe': severity_proportions_severe,
    'Viv': severity_proportions_viv,
    'Raddad': severity_proportions_raddad,
    'Robinson': severity_proportions_robinson,
}

annual_cases = 19202639

for scenario_name, scenario in severity_scenarios.items():
    total_burden = cb.calculate_severity_adjusted_burden(conservative_prevalence_merged, scenario, annual_cases)
    print(f"Total burden assuming all cases are {scenario_name.lower()}: {total_burden}")


# Plot symptom prevalence
plot_rp.plot_symptom_prevalence(conservative_prevalence_merged)


# Plot symptom prevalence by adjustment method
adjustment_scenarios = {'Moderate': moderate_prevalence, 'Conservative': conservative_prevalence_merged}
plot_rp.plot_symptoms_prevalence_across_scenarios(adjustment_scenarios)


# Plot total burden by severity scenario
total_burdens = [cb.calculate_severity_adjusted_burden(conservative_prevalence_merged, scenario, annual_cases) for scenario in severity_scenarios.values()]
plot_rp.plot_total_burden_under_different_severities(severity_scenarios, total_burdens)


# Plot symptom burden by multiple severity and prevalence scenarios
plot_rp.plot_symptom_burden_by_scenarios(adjustment_scenarios, severity_scenarios, annual_cases)


# Set-up PyMC model
time, prevalence, symptom_idx = ed.prepare_data(conservative_prevalence)
model = ed.setup_model(
    time, 
    prevalence, 
    symptom_idx, 
    n_symptoms=len(conservative_prevalence['symptom'].unique())
)

# Prior predictive check
time_points = np.linspace(0, 18, 300)  # Adjust as needed (0 to 18 months in this example)
plot_pd.plot_implied_paths(model, time_points, plot_bands=False)

# Posterior predictive check
trace = ed.sample_model(model)
symptom_names = conservative_prevalence['symptom'].unique().tolist()
plot_pd.plot_all_symptoms(trace, symptom_names, time_points=np.linspace(0, 18, 100))

# Plot per-symptom and overall burden distributions
num_samples=3000
annual_cases = sq.to(7000000, 40000000, credibility=99) @ num_samples

burdens = plot_pd.calculate_time_adjusted_burden_with_uncertainty(model, severity_proportions_mild, annual_cases, data_daly)

plot_pd.plot_burden_distributions(burdens)
