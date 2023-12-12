import prepare as prep
import adjust_prevalence as ap
import calculate_burden as cb
import plot as plot
import pandas as pd

# Load data from CSV files
prevalence_diff = prep.prepare_raw_data()
data_daly = pd.read_csv('data/daly.csv')

# Adjusting the prevalence data
conservative_prevalence = ap.adjust_prevalence_data(prevalence_diff, method='conservative')
moderate_prevalence = ap.adjust_prevalence_data(prevalence_diff, method='moderate')

# Merging data
merged_data = cb.merge_prevalence_severity(conservative_prevalence, data_daly)
merged_data = cb.calculate_period_burden(merged_data)

# Define variables
annual_cases = 19202639

severity_proportions_mild = {'mild': 1, 'moderate': 0, 'severe': 0}
severity_proportions_moderate = {'mild': 0, 'moderate': 1, 'severe': 0}
severity_proportions_severe = {'mild': 0, 'moderate': 0, 'severe': 1}
severity_proportions_viv = {'mild': 0.9199, 'moderate': 0.0714, 'severe': 0.0088}
severity_proportions_raddad = {'mild': 0.9975, 'moderate': 0.0023, 'severe': 0.0002}
severity_proportions_robinson = {'mild': 0.94, 'moderate': 0.047, 'severe': 0.013}

total_burden_mild = cb.calculate_severity_adjusted_burden(merged_data, severity_proportions_mild, annual_cases)
total_burden_moderate = cb.calculate_severity_adjusted_burden(merged_data, severity_proportions_moderate, annual_cases)
total_burden_severe = cb.calculate_severity_adjusted_burden(merged_data, severity_proportions_severe, annual_cases)
total_burden_viv = cb.calculate_severity_adjusted_burden(merged_data, severity_proportions_viv, annual_cases)
total_burden_raddad = cb.calculate_severity_adjusted_burden(merged_data, severity_proportions_raddad, annual_cases)
total_burden_robinson = cb.calculate_severity_adjusted_burden(merged_data, severity_proportions_robinson, annual_cases)

print(f"Total burden assuming all cases are mild, moderate, or severe respectively: {total_burden_mild}, {total_burden_moderate}, {total_burden_severe}")
print(f"Total burden following Viv's COVID severity proportions: {total_burden_viv}")
print(f"Total burden following Raddad's COVID severity proportions: {total_burden_raddad}")
print(f"Total burden following Robinson's COVID severity proportions: {total_burden_robinson}")


# Plot symptom prevalence
plot.plot_all_symptoms_prevalence(merged_data)


# Plot symptom prevalence by adjustment method
scenarios = {'Moderate': moderate_prevalence, 'Conservative': conservative_prevalence}

plot.plot_all_symptoms_prevalence_across_scenarios(prevalence_diff, moderate_prevalence, conservative_prevalence)


# Plot total burden by severity scenario
severity_scenarios = {
    'Mild': severity_proportions_mild,
    'Viv': severity_proportions_viv,
    'Raddad': severity_proportions_raddad,
    'Robinson': severity_proportions_robinson,
}
total_burdens = [cb.calculate_severity_adjusted_burden(merged_data, scenario, annual_cases) for scenario in severity_scenarios.values()]
plot.plot_total_burden_under_different_severities(merged_data, severity_scenarios, annual_cases)


# Plot symptom burden by multiple severity and prevalence scenarios
severity_scenarios = [
    severity_proportions_mild,
    severity_proportions_viv,
    severity_proportions_raddad,
    severity_proportions_robinson
]
prevalence_methods = [
    cb.adjust_prevalence_data(method='conservative'), 
    cb.adjust_prevalence_data(method='moderate')
    ]

plot.plot_multiple_scenarios(merged_data, severity_scenarios, annual_cases, prevalence_methods)

