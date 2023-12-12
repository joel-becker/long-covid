import pandas as pd

def merge_prevalence_severity(prevalence_data, severity_data):
    """
    Merges prevalence and severity data.

    Parameters:
    prevalence_data (pd.DataFrame): DataFrame containing prevalence data.
    severity_data (pd.DataFrame): DataFrame containing severity data.

    Returns:
    pd.DataFrame: Merged data.
    """
    return pd.merge(prevalence_data, severity_data, left_on='symptom', right_on='symptom')

def calculate_period_burden(merged_data):
    """
    Calculates the burden for different periods.

    Parameters:
    merged_data (pd.DataFrame): Merged data containing prevalence and severity data.

    Returns:
    pd.DataFrame: Data with additional columns for burden in different periods.
    """
    time_ranges = {'6m': (2, 8), '12m': (8, 15), '18m': (15, 36)}

    for period, (start_month, end_month) in time_ranges.items():
        proportion_of_year = (end_month - start_month) / 12
        prevalence_col = f'prevalence_diff_{period}'
        burden_col = f'extra_burden_{period}'
        merged_data[burden_col] = merged_data[prevalence_col] * merged_data['daly_adjustment'] * proportion_of_year

    return merged_data

def calculate_severity_adjusted_burden(merged_data, severity_proportions, total_cases):
    """
    Calculates the severity-adjusted burden.

    Parameters:
    merged_data (pd.DataFrame): Merged data containing prevalence and severity data.
    severity_proportions (dict): Proportions for severity levels (mild, moderate, severe).
    total_cases (int): Total number of cases.

    Returns:
    float: Total severity-adjusted burden.
    """
    aggregate_burden = 0
    for period in ['6m', '12m', '18m']:
        burden_col = f'extra_burden_{period}'
        for severity, proportion in severity_proportions.items():
            severity_burden = merged_data[merged_data[severity] == 1][burden_col].sum()
            aggregate_burden += severity_burden * proportion * total_cases

    return aggregate_burden
