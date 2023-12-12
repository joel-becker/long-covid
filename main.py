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
conservative_prevalence = cb.merge_prevalence_severity(conservative_prevalence, data_daly)
conservative_prevalence = cb.calculate_period_burden(conservative_prevalence)

moderate_prevalence = cb.merge_prevalence_severity(moderate_prevalence, data_daly)
moderate_prevalence = cb.calculate_period_burden(moderate_prevalence)


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
    total_burden = cb.calculate_severity_adjusted_burden(conservative_prevalence, scenario, annual_cases)
    print(f"Total burden assuming all cases are {scenario_name.lower()}: {total_burden}")


# Plot symptom prevalence
plot.plot_symptom_prevalence(conservative_prevalence)


# Plot symptom prevalence by adjustment method
adjustment_scenarios = {'Moderate': moderate_prevalence, 'Conservative': conservative_prevalence}
plot.plot_symptoms_prevalence_across_scenarios(adjustment_scenarios)


# Plot total burden by severity scenario
total_burdens = [cb.calculate_severity_adjusted_burden(conservative_prevalence, scenario, annual_cases) for scenario in severity_scenarios.values()]
plot.plot_total_burden_under_different_severities(severity_scenarios, total_burdens)


# Plot symptom burden by multiple severity and prevalence scenarios
plot.plot_symptom_burden_by_scenarios(adjustment_scenarios, severity_scenarios, annual_cases)

