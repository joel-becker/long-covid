import prepare as prep
import adjust_prevalence as ap
import calculate_burden as cb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load data from CSV files
prevalence_diff = prep.prepare_raw_data()
data_daly = pd.read_csv('data/daly.csv')

# Adjusting the prevalence data
adjusted_prevalence = ap.adjust_prevalence_data(prevalence_diff, method='conservative')

# Merging data
merged_data = cb.merge_prevalence_severity(adjusted_prevalence, data_daly)
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










def plot_all_symptoms_prevalence(merged_data):
    # Define the periods and corresponding months
    periods_months = {
        '6m': (2, 8),
        '12m': (8, 15),
        '18m': (15, 36)
    }
    
    # Initialize the plot
    plt.figure(figsize=(14, 7))
    
    # Iterate over each symptom
    for symptom in merged_data['symptom'].unique():
        # Extract the symptom-specific data
        symptom_data = merged_data[merged_data['symptom'] == symptom].iloc[0]
        x_values = [0]  # Start from month 0
        y_values = [0]  # Start from prevalence 0
        
        # For each period, retrieve the prevalence and calculate the x-axis values
        for period, months in periods_months.items():
            prevalence = symptom_data[f'prevalence_diff_{period}']
            # Extend the flat line for the period
            x_values.extend([months[0], months[1]])
            y_values.extend([prevalence, prevalence])
        
        # Append the closing points of the plot (back to zero)
        x_values.append(36)
        y_values.append(0)
        x_values.append(48)
        y_values.append(0)
        
        # Plot the data
        plt.plot(x_values, y_values, label=symptom)
    
    # Customize the plot
    plt.title('Prevalence of Long COVID Symptoms Over Time')
    plt.xlabel('Months Since Infection')
    plt.ylabel('Prevalence (%)')
    plt.xticks([0, 2, 8, 15, 36, 48], ['0', '2', '8', '15', '36', '>36'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Remove the plot frame lines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Run the function to generate the plot
plot_all_symptoms_prevalence(merged_data)


import matplotlib.pyplot as plt

def plot_all_symptoms_prevalence_across_scenarios(merged_data, moderate_data, conservative_data):
    # Define the periods and corresponding months
    periods_months = {
        '6m': (2, 8),
        '12m': (8, 15),
        '18m': (15, 36)
    }
    
    # Initialize the plot with horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    scenarios = {'Original': merged_data, 'Moderate': moderate_data, 'Conservative': conservative_data}

    for ax, (scenario, data) in zip(axes, scenarios.items()):
        # Iterate over each symptom for the scenario
        for symptom in data['symptom'].unique():
            # Extract the symptom-specific data
            symptom_data = data[data['symptom'] == symptom].iloc[0]
            x_values = [0]  # Start from month 0
            y_values = [0]  # Start from prevalence 0
            
            # For each period, retrieve the prevalence and calculate the x-axis values
            for period, months in periods_months.items():
                prevalence = symptom_data[f'prevalence_diff_{period}']
                # Extend the flat line for the period
                x_values.extend([months[0], months[1]])
                y_values.extend([prevalence, prevalence])
            
            # Append the closing points of the plot (back to zero)
            x_values.append(36)
            y_values.append(0)
            x_values.append(48)
            y_values.append(0)
            
            # Plot the data for the symptom in the current scenario
            ax.plot(x_values, y_values, label=symptom)

        # Customize the plot for the scenario
        ax.set_title(f'Prevalence Scenario: {scenario}')
        ax.set_xlabel('Months Since Infection')
        ax.grid(True, alpha=0.3)

        # Remove the plot frame lines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Common settings
    plt.ylabel('Prevalence (%)')
    plt.yticks([0, 5, 10, 15], ['0%', '5%', '10%', '15%'])
    axes[0].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Usage example (assuming original_data, conservative_data, and moderate_data are defined):
plot_all_symptoms_prevalence_across_scenarios(prevalence_diff, moderate_prevalence_diff, conservative_prevalence_diff)




def plot_total_burden_under_different_severities(merged_data, severity_scenarios, total_cases):
    """
    Plots the total DALYs lost under different severity scenarios.
    """
    total_burdens = []

    # Calculate total burden for each scenario
    for scenario in severity_scenarios:
        total_burdens.append(cb.calculate_severity_adjusted_burden(merged_data, severity_scenarios[scenario], total_cases))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(severity_scenarios.keys(), total_burdens, color='skyblue')
    ax.set_xlabel('Severity Scenarios')
    ax.set_ylabel('Total DALYs Lost')
    ax.set_title('Total Burden Under Different Severity Scenarios')
    ax.set_xticklabels(severity_scenarios.keys(), rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Tufte-style minimalism
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('gray')

    # Comma separation for y-axis labels
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.show()

severity_scenarios = {
    'Viv': severity_proportions_viv,
    'Raddad': severity_proportions_raddad,
    'Robinson': severity_proportions_robinson,
}

plot_total_burden_under_different_severities(merged_data, severity_scenarios, annual_cases)



def calculate_weighted_burden_by_symptom(merged_data, severity_proportions, total_cases, prevalence_method):
    # Adjust prevalence based on the method
    adjusted_data = prevalence_method(merged_data)  # Assuming prevalence_method is a function like conservative_adjustment
    severity_levels = ['mild', 'moderate', 'severe']
    burden_by_symptom = {}

    for symptom in adjusted_data['symptom'].unique():
        symptom_data = adjusted_data[adjusted_data['symptom'] == symptom]
        total_symptom_burden = 0

        for severity in severity_levels:
            for period in ['6m', '12m', '18m']:
                burden_col = f'extra_burden_{period}'
                # Calculate the burden for the severity level
                severity_burden = symptom_data[symptom_data[severity] == 1][burden_col].sum()
                
                # Adjust based on severity proportions
                severity_adjusted_burden = severity_burden * severity_proportions[severity]
    
                # Aggregate the burden across the population
                total_symptom_burden += severity_adjusted_burden * total_cases

        burden_by_symptom[symptom] = total_symptom_burden

    return burden_by_symptom

def plot_multiple_scenarios(merged_data, severity_scenarios, total_cases, prevalence_methods):
    num_scenarios = len(severity_scenarios) * len(prevalence_methods)
    subplot_rows = len(severity_scenarios)
    subplot_cols = len(prevalence_methods)
    max_burden = 0

    # Pre-calculate maximum burden for consistent x-axis
    for severity_proportions in severity_scenarios:
        for prevalence_method in prevalence_methods:
            burden_by_symptom = calculate_weighted_burden_by_symptom(merged_data, severity_proportions, total_cases, prevalence_method)
            max_burden = max(max_burden, max(burden_by_symptom.values()))

    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=(12, 6 * subplot_rows), sharex=True, sharey=True)

    for i, severity_proportions in enumerate(severity_scenarios):
        for j, prevalence_method in enumerate(prevalence_methods):
            ax = axes[i, j] if subplot_rows > 1 else axes[j]
            burden_by_symptom = calculate_weighted_burden_by_symptom(merged_data, severity_proportions, total_cases, prevalence_method)
            
            # Create DataFrame for plotting
            plot_data = pd.DataFrame(list(burden_by_symptom.items()), columns=['Symptom', 'Burden'])
            plot_data.sort_values(by='Burden', ascending=False, inplace=True)

            # Plotting
            ax.barh(plot_data['Symptom'], plot_data['Burden'], color='skyblue')
            ax.set_xlim(0, max_burden * 1.1)  # Slightly above the max to ensure visibility

            # Setting up titles and axis labels
            if j == 0:
                ax.set_ylabel('Symptoms')
            if i == subplot_rows - 1:
                ax.set_xlabel('Total Burden (DALYs)')
            ax.set_title(f'{severity_proportions}, {prevalence_method.__name__}')

            # Reducing borders and non-data ink
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.spines['left'].set_visible(i == 0)
            #ax.spines['bottom'].set_visible(i == subplot_rows - 1)

            # Comma separation for y-axis labels
            ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

            # Ticks only on necessary axes
            ax.xaxis.set_tick_params(labelbottom=(i == subplot_rows - 1))
            ax.yaxis.set_tick_params(labelleft=(j == 0))

    plt.tight_layout()
    plt.show()



severity_scenarios = [
    #severity_proportions_mild
    #severity_proportions_moderate
    #severity_proportions_severe
    severity_proportions_viv,
    severity_proportions_raddad,
    severity_proportions_robinson
]
prevalence_methods = [conservative_adjustment, moderate_adjustment]

plot_multiple_scenarios(merged_data, severity_scenarios, annual_cases, prevalence_methods)

