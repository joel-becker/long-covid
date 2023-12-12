def adjust_prevalence_data(prevalence_data, method='conservative'):
    """
    Adjusts the prevalence data based on the chosen method.
    
    Parameters:
    prevalence_data (pd.DataFrame): DataFrame containing prevalence data.
    method (str): Adjustment method, either 'conservative' or 'moderate'.
    
    Returns:
    pd.DataFrame: Adjusted prevalence data.
    """
    adjusted = prevalence_data.copy()

    # Columns representing different time periods
    col_6m, col_12m, col_18m = 'prevalence_diff_6m', 'prevalence_diff_12m', 'prevalence_diff_18m'

    if method == 'conservative':
        adjusted = _conservative_adjustment(adjusted, col_6m, col_12m, col_18m)
    elif method == 'moderate':
        adjusted = _moderate_adjustment(adjusted, col_6m, col_12m, col_18m)

    return adjusted


def _conservative_adjustment(data, col_6m, col_12m, col_18m):
    # Adjust 18-month based on 12-month, then adjust 12-month based on 6-month, and re-adjust 18-month
    for col_higher, col_lower in [(col_18m, col_12m), (col_12m, col_6m), (col_18m, col_12m)]:
        data[col_higher] = data[[col_lower, col_higher]].min(axis=1)
    return data


def _moderate_adjustment(data, col_6m, col_12m, col_18m):
    # Create a mean of all three columns and adjust non-decreasing trends
    data['mean_all'] = data[[col_6m, col_12m, col_18m]].mean(axis=1)
    is_non_decreasing = (data[col_12m] >= data[col_6m]) & (data[col_18m] >= data[col_12m])
    data.loc[is_non_decreasing, [col_6m, col_12m, col_18m]] = data['mean_all']

    # Ensure non-decreasing trend for remaining cases
    for pair in [(col_12m, col_18m), (col_6m, col_12m)]:
        mean_col = f'mean_{"_".join(pair)}'
        data[mean_col] = data[pair].mean(axis=1)
        data.loc[data[pair[1]] >= data[pair[0]], pair] = data[mean_col]

    # Re-check and adjust for non-decreasing trend between 12 and 18 months
    data.loc[data[col_18m] > data[col_12m], [col_6m, col_12m, col_18m]] = data['mean_all']

    # Drop the temporary mean columns
    data.drop(columns=['mean_all', 'mean_12_18', 'mean_6_12'], inplace=True)
    return data
