def wrangle_daly_data(data_daly):
    """
    Wrangles the data_daly DataFrame into a format with symptom name, "GHE2019", 
    and mild, moderate, and severe columns.

    Parameters:
    data_daly (pd.DataFrame): The original DALY data DataFrame.

    Returns:
    pd.DataFrame: A wrangled DataFrame with selected columns.
    """
    data_daly['symptom'] = data_daly['name_merge_data']
    data_daly['daly_adjustment'] = data_daly['GHE2019']
    required_columns = ['symptom', 'daly_adjustment', 'mild', 'moderate', 'severe']
    
    # Ensure all required columns are present
    if not all(col in data_daly.columns for col in required_columns):
        raise ValueError("One or more required columns are missing from the data_daly DataFrame.")

    return data_daly[required_columns]

