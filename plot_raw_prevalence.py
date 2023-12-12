import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

def plot_symptom_prevalence(merged_data, title_suffix=''):
    # Define the periods and corresponding months
    periods_months = {'6m': (2, 8), '12m': (8, 15), '18m': (15, 36)}
    
    # Initialize the plot
    plt.figure(figsize=(14, 7))
    
    # Iterate over each symptom
    for symptom in merged_data['symptom'].unique():
        # Extract the symptom-specific data
        symptom_data, x_values, y_values = merged_data[merged_data['symptom'] == symptom].iloc[0], [0], [0]
        
        # For each period, retrieve the prevalence and calculate the x-axis values
        for period, months in periods_months.items():
            prevalence = symptom_data[f'prevalence_diff_{period}']

            # Extend the flat line for the period
            x_values.extend([months[0], months[1]])
            y_values.extend([prevalence, prevalence])
        
        # Append the closing points of the plot (back to zero)
        x_values += [36, 48]
        y_values += [0, 0]

        # Plot the data
        plt.plot(x_values, y_values, label=symptom)
    
    # Customize the plot
    _style_plot(title=f'Prevalence of Symptoms Over Time {title_suffix}', xlabel='Months Since Infection', ylabel='Prevalence (%)')
    plt.xticks([0, 2, 8, 15, 36, 48], ['0', '2', '8', '15', '36', '>36'])
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_symptoms_prevalence_across_scenarios(scenarios):
    # Define the periods and corresponding months
    periods_months = {'6m': (2, 8), '12m': (8, 15), '18m': (15, 36)}
    
    # Initialize the plot with horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax, (scenario_name, data) in zip(axes, scenarios.items()):
        # Iterate over each symptom for the scenario
        for symptom in data['symptom'].unique():
            # Extract the symptom-specific data
            symptom_data, x_values, y_values = data[data['symptom'] == symptom].iloc[0], [0], [0]
            
            # For each period, retrieve the prevalence and calculate the x-axis values
            for period, months in periods_months.items():
                prevalence = symptom_data[f'prevalence_diff_{period}']

                # Extend the flat line for the period
                x_values.extend([months[0], months[1]])
                y_values.extend([prevalence, prevalence])
            
            # Append the closing points of the plot (back to zero)
            x_values += [36, 48]
            y_values += [0, 0]
            
            # Plot the data for the symptom in the current scenario
            ax.plot(x_values, y_values, label=symptom)
        
        # Customize the plot for the scenario
        _style_subplot(ax, title=f'Prevalence Scenario: {scenario_name}', xlabel='Months Since Infection')
    
    # Common settings
    plt.ylabel('Prevalence (%)')
    plt.yticks([0, 5, 10, 15], ['0%', '5%', '10%', '15%'])
    axes[0].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_total_burden_under_different_severities(severity_scenarios, total_burdens):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(severity_scenarios.keys(), total_burdens, color='skyblue')
    _style_plot(title='Total Burden Under Different Severity Scenarios', xlabel='Severity Scenarios', ylabel='Total DALYs Lost')
    
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_tick_params(rotation=45)

    # Comma separation for y-axis labels
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.show()

def plot_symptom_burden_by_scenarios(adjustment_scenarios, severity_scenarios, total_cases):
    max_burden = 0

    # Calculate the maximum burden for consistent x-axis scaling
    for adj_name, adj_data in adjustment_scenarios.items():
        for sev_name, sev_props in severity_scenarios.items():
            burdens = calculate_weighted_burden_by_symptom(adj_data, sev_props, total_cases)
            max_burden = max(max_burden, max(burdens.values()))

    fig, axes = plt.subplots(len(severity_scenarios), len(adjustment_scenarios), figsize=(12, 6 * len(severity_scenarios)), sharex=True, sharey=True)

    for i, (sev_name, sev_props) in enumerate(severity_scenarios.items()):
        for j, (adj_name, adj_data) in enumerate(adjustment_scenarios.items()):
            ax = axes[i][j] if len(severity_scenarios) > 1 else axes[j]
            burdens = calculate_weighted_burden_by_symptom(adj_data, sev_props, total_cases)

            plot_data = pd.DataFrame(list(burdens.items()), columns=['Symptom', 'Burden']).sort_values(by='Burden', ascending=False)
            ax.barh(plot_data['Symptom'], plot_data['Burden'], color='skyblue')
            ax.set_xlim(0, max_burden * 1.1)

            if j == 0:
                ax.set_ylabel('Symptoms')
            if i == len(severity_scenarios) - 1:
                ax.set_xlabel('Total Burden (DALYs)')
            ax.set_title(f'{sev_name}, {adj_name}')
            _style_subplot(ax, f'{sev_name}, {adj_name}', show_spines=False, show_grid=False)
    
    plt.tight_layout()
    plt.show()


def _style_plot(title='', xlabel='', ylabel=''):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    _hide_spines()

def _style_subplot(ax, title='', xlabel='', show_spines=True, show_grid=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(show_grid, alpha=0.3)
    if not show_spines:
        _hide_spines(ax)

def _hide_spines(ax=None):
    if ax is None:
        ax = plt.gca()
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

def calculate_weighted_burden_by_symptom(adjusted_data, severity_proportions, total_cases):
    # Adjust prevalence based on the method
    burden_by_symptom = {}
    for symptom in adjusted_data['symptom'].unique():
        symptom_data = adjusted_data[adjusted_data['symptom'] == symptom]
        total_symptom_burden = 0

        for severity in ['mild', 'moderate', 'severe']:
            for period in ['6m', '12m', '18m']:
                # Calculate the burden for the severity level
                severity_burden = symptom_data[symptom_data[severity] == 1][f'extra_burden_{period}'].sum()
                
                # Adjust based on severity proportions
                severity_adjusted_burden = severity_burden * severity_proportions[severity]
    
                # Aggregate the burden across the population
                total_symptom_burden += severity_adjusted_burden * total_cases

        burden_by_symptom[symptom] = total_symptom_burden
    return burden_by_symptom
