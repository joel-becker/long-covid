import squigglepy as sq
import pandas as pd

param_descriptions = {
    'size': 'Total population size',
    'baseline_risk': 'Baseline risk of long COVID',
    'infection_rate': 'Weekly infection rate',
    'strain_reduction_factor': 'Reduction factor for new strains',
    'total_strains': 'Number of distinct strains', 
    'current_strain': 'Current strain', 
    'strain_decay': 'Half-life of strain relative prevalence',
    'initial_vaccination_distribution': 'Initial distribution of number of vaccinations per person',
    'vaccination_reduction': 'Peak effectiveness of vaccination (against Long COVID, conditional on COVID)', 
    'vaccination_interval': 'Minimum interval between vaccinations', 
    'vaccination_effectiveness_halflife': 'Half-life of vaccination effectiveness', 
    'vaccination_hazard_rate': 'Hazard rate of receiving vaccination',
    'aor_value': 'AOR value'
}

default_params = {
    'size': 330_000,
    'baseline_risk': sq.norm(mean = 0.15, sd = 0.01), 
    'infection_rate': sq.norm(mean=(19/330)/52, sd=0.0001),
    'strain_reduction_factor': sq.norm(mean=0.6, sd=0.1), 
    'total_strains': 10, 
    'current_strain': 1, 
    'strain_decay': 50,
    'initial_vaccination_distribution': {0: 0.2, 1: 0.2, 2: 0.3, 3:0.2, 4:0.1},
    'vaccination_reduction': sq.beta(100*0.25, 100*(1-0.25)), 
    'vaccination_interval': 180, 
    'vaccination_effectiveness_halflife': sq.norm(mean=235, sd=30), 
    'vaccination_hazard_rate': sq.beta(1000*0.01, 1000*(1-0.01)),
    'aor_value': sq.beta(100*0.72, 100*(1-0.72))
    # Default values for other parameters
}

# Example scenario override
pessimistic_params = {
    'infection_rate': (25/330)/52,  # Higher infection rate in pessimistic scenario
    # Other overrides
}

def merge_and_describe_parameters(default_params, scenario_params, descriptions):
    # Merge default and scenario-specific parameters
    merged_params = {**default_params, **scenario_params}
    
    # Prepare data for DataFrame
    data = []
    for param, value in merged_params.items():
        description = descriptions.get(param, "No description available")
        data.append([param, description, value])
    
    return pd.DataFrame(data, columns=['Parameter', 'Description', 'Value'])


def is_number(value):
    """Check if the value is a number."""
    return isinstance(value, (int, float))

def format_distribution(value):
    """Format distribution objects based on numerical output."""
    # Attempt to access and format mean and standard deviation if applicable
    try:
        if is_number(value.mean) and is_number(value.sd):
            formatted_mean = f"{value.mean:.3g}"
            formatted_std = f"{value.sd:.3g}"
            return f"Normal({formatted_mean}, {formatted_std})"
    except AttributeError:
        pass

    # Attempt to format based on 'a' and 'b' attributes for Beta distributions
    try:
        if is_number(value.a) and is_number(value.b):
            formatted_a = f"{value.a:.3g}"
            formatted_b = f"{value.b:.3g}"
            return f"Beta({formatted_a}, {formatted_b})"
    except AttributeError:
        pass

    # If none of the above, return a generic representation
    return "Distribution"

def format_value(value):
    """Format numbers, distributions, and dictionaries appropriately."""
    if is_number(value):  # Direct numerical values
        return f"{value:.3g}"
    elif isinstance(value, dict):  # Dictionaries
        return {k: format_value(v) for k, v in value.items()}
    else:  # Attempt to format as distribution
        return format_distribution(value)

def generate_comparison_table(default_params, scenario_params, descriptions):
    # Create an empty DataFrame for the comparison table
    table = pd.DataFrame(columns=['Parameter', 'Description', 'Mainline', 'Pessimistic'])
    
    # Iterate over all parameters in the descriptions
    rows = []
    for param, description in descriptions.items():
        mainline_value = format_value(default_params.get(param, 'N/A'))
        pessimistic_value = format_value(scenario_params.get(param, default_params.get(param, 'N/A')))
        rows.append({'Parameter': param, 'Description': description, 'Mainline': mainline_value, 'Pessimistic': pessimistic_value})
    
    # Use pd.concat instead of append to avoid FutureWarning
    table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    
    return table
