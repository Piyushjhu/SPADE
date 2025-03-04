# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Constants
density_copper = 8950  # kg/m^3
acoustic_velocity_copper = 3950  # m/s

# Load both datasets
datasets_4um = {
    '800mJ': pd.read_csv('4um_spall_strength_strain_rate_table_800mJ.csv'),
    '1200mJ': pd.read_csv('4um_spall_strength_strain_rate_table_1200mJ.csv'),
    '1350mJ': pd.read_csv('4um_spall_strength_strain_rate_table_1350mJ.csv'),
    '1500mJ': pd.read_csv('4um_spall_strength_strain_rate_table_1500mJ.csv')
}

datasets_100nm = {
    '800mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_800mJ.csv'),
    '1000mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_1000mJ.csv'),
    '1200mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_1200mJ.csv'),
    '1350mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_1350mJ.csv'),
    '1500mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_1500mJ.csv'),
    '1600mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_1600mJ.csv')
}

# Calculate Peak Shock Pressure for 100nm dataset (already exists)
for dataset in datasets_100nm.values():
    dataset['Peak Shock Pressure (GPa)'] = dataset['First Maxima (m/s)'] * 0.5 * density_copper * acoustic_velocity_copper * 1e-9

# Calculate Peak Shock Pressure for 4um dataset (missing)
for dataset in datasets_4um.values():
    dataset['Peak Shock Pressure (GPa)'] = dataset['First Maxima (m/s)'] * 0.5 * density_copper * acoustic_velocity_copper * 1e-9

# Function to preprocess and filter data
def preprocess_data(datasets, grain_size):
    all_data = pd.concat(datasets.values(), ignore_index=True)
    all_data['Grain Size (µm)'] = grain_size  # Add the grain size as a new column
    all_data = all_data.dropna(subset=['Spall Strength (GPa)', 'Strain Rate (s^-1)', 'Peak Shock Pressure (GPa)'])
    return all_data

# Preprocess both datasets
data_4um = preprocess_data(datasets_4um, 4)  # Grain size 4 µm
data_100nm = preprocess_data(datasets_100nm, 0.1)  # Grain size 100 nm

# Plot sensitivity analysis for strain rate, shock stress, and grain size
def plot_sensitivity_analysis(data_4um, data_100nm):
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Scatter plots for 4 µm data
    sns.scatterplot(ax=axes[0], data=data_4um, x='Strain Rate (s^-1)', y='Spall Strength (GPa)', color='b', label='4 µm')
    axes[0].set_title('Strain Rate vs Spall Strength (4 µm)')
    axes[0].set_xlabel('Strain Rate (s^-1)')
    axes[0].set_ylabel('Spall Strength (GPa)')

    # Scatter plots for 100 nm data
    sns.scatterplot(ax=axes[1], data=data_100nm, x='Strain Rate (s^-1)', y='Spall Strength (GPa)', color='r', label='100 nm')
    axes[1].set_title('Strain Rate vs Spall Strength (100 nm)')
    axes[1].set_xlabel('Strain Rate (s^-1)')
    axes[1].set_ylabel('Spall Strength (GPa)')

    # Combine both datasets for a broader view
    combined_data = pd.concat([data_4um[['Strain Rate (s^-1)', 'Spall Strength (GPa)']],
                                data_100nm[['Strain Rate (s^-1)', 'Spall Strength (GPa)']]], 
                                ignore_index=True)
    # combined_data['Dataset'] = ['4 µm' if i < len(data_4um) else '100 nm' for i in range(len(combined_data))]
    # sns.scatterplot(ax=axes[2], data=combined_data, x='Strain Rate (s^-1)', y='Spall Strength (GPa)', hue='Dataset', palette="coolwarm")
    
    sns.scatterplot(ax=axes[2], data=combined_data, x='Strain Rate (s^-1)', y='Spall Strength (GPa)', hue=combined_data.index // len(data_4um), palette="coolwarm")
    axes[2].set_title('Strain Rate vs Spall Strength (Combined)')
    axes[2].set_xlabel('Strain Rate (s^-1)')
    axes[2].set_ylabel('Spall Strength (GPa)')

    plt.tight_layout()
    plt.show()

# Run sensitivity analysis plot
plot_sensitivity_analysis(data_4um, data_100nm)

# Sensitivity analysis using linear regression for strain rate
def linear_regression_analysis(data_4um, data_100nm):
    X_4um = data_4um[['Strain Rate (s^-1)', 'Peak Shock Pressure (GPa)', 'Grain Size (µm)']]
    y_4um = data_4um['Spall Strength (GPa)']
    X_100nm = data_100nm[['Strain Rate (s^-1)', 'Peak Shock Pressure (GPa)', 'Grain Size (µm)']]
    y_100nm = data_100nm['Spall Strength (GPa)']
    
    # Standardize the data
    scaler = StandardScaler()
    X_4um_scaled = scaler.fit_transform(X_4um)
    X_100nm_scaled = scaler.transform(X_100nm)
    
    # Train Linear Regression Models
    model_4um = LinearRegression().fit(X_4um_scaled, y_4um)
    model_100nm = LinearRegression().fit(X_100nm_scaled, y_100nm)
    
    # Plot regression coefficients
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.bar(['Strain Rate (s^-1)', 'Shock Pressure (GPa)', 'Grain Size (µm)'],
           model_4um.coef_, color='b', alpha=0.5, label="4 µm Model Coefficients")
    ax.bar(['Strain Rate (s^-1)', 'Shock Pressure (GPa)', 'Grain Size (µm)'],
           model_100nm.coef_, color='r', alpha=0.5, label="100 nm Model Coefficients")
    
    ax.set_title('Sensitivity of Spall Strength to Variables (Linear Regression)')
    ax.set_ylabel('Coefficient Value')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Run linear regression sensitivity analysis
linear_regression_analysis(data_4um, data_100nm)
# %%
