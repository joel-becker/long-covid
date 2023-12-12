import pandas as pd

def clean_and_subset_data(data):
    # Subset relevant columns and remove NA values
    return data[['symptomatic', 'cohort_period', 'symptom', 'percentage_1st_period', 'percentage_2nd_period']].dropna()

def calculate_prevalence_differences(cleaned_data):
    # Group by symptom and cohort period
    grouped = cleaned_data.groupby(['symptom', 'cohort_period'])

    # Calculate prevalence differences for the first and second periods (6 months, 12 or 18 months)
    prevalence_diff_6m = grouped.apply(lambda x: calculate_mean_diff(x, 'percentage_1st_period')).reset_index(name='prevalence_diff_6m')
    prevalence_diff_2nd = grouped.apply(lambda x: calculate_mean_diff(x, 'percentage_2nd_period')).reset_index(name='prevalence_diff_2nd')

    return prevalence_diff_6m, prevalence_diff_2nd

def calculate_mean_diff(group, column):
    return group[group['symptomatic'] == 1][column].mean() - group[group['symptomatic'] == 0][column].mean()

def merge_prevalence_data(prevalence_diff_6m, prevalence_diff_2nd):
    # Merge the dataframes
    merged = pd.merge(prevalence_diff_6m, prevalence_diff_2nd, on=['symptom', 'cohort_period'])

    # Add a column to determine the second period's duration based on the cohort period string
    merged['months_2nd_period'] = merged['cohort_period'].apply(lambda x: 18 if '18' in x else 12)
    return merged

def collapse_prevalence_data(merged_data):
   # Calculate the mean prevalence difference at 6 months for each symptom
    mean_prevalence_diff_6m = merged_data.groupby('symptom')['prevalence_diff_6m'].mean().reset_index()

    # Update the original dataframe with these mean values
    merged_data = merged_data.merge(mean_prevalence_diff_6m, on='symptom', suffixes=('', '_mean'))

    # Replace the original 6-month prevalence differences with the mean values
    merged_data['prevalence_diff_6m'] = merged_data['prevalence_diff_6m_mean']
    merged_data = separate_and_drop_columns(merged_data)

    # Collapse rows so each symptom has one row with 6, 12, and 18 months data
    collapsed = merged_data.groupby('symptom').agg({'prevalence_diff_6m': 'first', 'prevalence_diff_12m': 'first', 'prevalence_diff_18m': 'first'}).reset_index()
    
    return collapsed

def separate_and_drop_columns(merged_data):
    # Create columns for 12 and 18 months prevalence differences
    merged_data['prevalence_diff_12m'] = merged_data.apply(lambda row: row['prevalence_diff_2nd'] if row['months_2nd_period'] == 12 else None, axis=1)
    merged_data['prevalence_diff_18m'] = merged_data.apply(lambda row: row['prevalence_diff_2nd'] if row['months_2nd_period'] == 18 else None, axis=1)

    # Drop columns that are no longer needed
    return merged_data.drop(columns=['cohort_period', 'months_2nd_period', 'prevalence_diff_2nd', 'prevalence_diff_6m_mean'])

def prepare_raw_data(prevalence_and_symptoms_file='data/prevalence_and_symptoms.csv'):
    # Load data from CSV files
    data_prevalence_and_symptoms = pd.read_csv(prevalence_and_symptoms_file)

    cleaned_data = clean_and_subset_data(data_prevalence_and_symptoms)
    prevalence_diff_6m, prevalence_diff_2nd = calculate_prevalence_differences(cleaned_data)
    merged_data = merge_prevalence_data(prevalence_diff_6m, prevalence_diff_2nd)
    collapsed_data = collapse_prevalence_data(merged_data)

    return collapsed_data
