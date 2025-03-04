# %%
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

# Constants
density_copper = 8950  # kg/m^3
acoustic_velocity_copper = 3950  # m/s

# Preprocess data
def preprocess_data_with_errors(datasets, grain_size):
    all_data = pd.concat(datasets.values(), ignore_index=True)
    all_data['Grain Size (µm)'] = grain_size  # Add the grain size as a new column
    all_data = all_data.dropna(subset=['Spall Strength (GPa)', 'Strain Rate (s^-1)', 'Peak Shock Pressure (GPa)'])
    return all_data

# Load the datasets (same process as before)
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

# Preprocess both datasets (grain size 4 µm and 100 nm)
data_4um = preprocess_data_with_errors(datasets_4um, 4)  # Grain size 4 µm
data_100nm = preprocess_data_with_errors(datasets_100nm, 0.1)  # Grain size 100 nm

# Use Strain Rate, Peak Shock Pressure, and Grain Size to predict Spall Strength
# We will model the uncertainty using the error values

X_4um = data_4um[['Strain Rate (s^-1)', 'Peak Shock Pressure (GPa)', 'Grain Size (µm)']].values
y_4um = data_4um['Spall Strength (GPa)'].values
y_err_4um = data_4um['Spall Strength Err (GPa)'].values  # Error in Spall Strength for 4 µm data

X_100nm = data_100nm[['Strain Rate (s^-1)', 'Peak Shock Pressure (GPa)', 'Grain Size (µm)']].values
y_100nm = data_100nm['Spall Strength (GPa)'].values
y_err_100nm = data_100nm['Spall Strength Err (GPa)'].values  # Error in Spall Strength for 100 nm data

# Function to perform Bayesian Linear Regression using PyMC4
def bayesian_linear_regression(X, y, y_err):
    with pm.Model() as model:
        # Priors for the coefficients
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        betas = pm.Normal('betas', mu=0, sigma=10, shape=X.shape[1])
        
        # Prior for the noise (error term)
        sigma = pm.HalfNormal('sigma', sigma=10)
        
        # Expected value of the outcome (linear regression model)
        mu = alpha + pm.math.dot(X, betas)
        
        # Likelihood function (the error term)
        likelihood = pm.Normal('y', mu=mu, sigma=y_err, observed=y)
        
        # Sampling from the posterior
        trace = pm.sample(2000, return_inferencedata=False)
    
    return trace

# Run Bayesian Linear Regression for both datasets (4 µm and 100 nm)
trace_4um = bayesian_linear_regression(X_4um, y_4um, y_err_4um)
trace_100nm = bayesian_linear_regression(X_100nm, y_100nm, y_err_100nm)

# # Plot the posterior distributions of the coefficients
# def plot_posterior(trace, varnames):
#     pm.traceplot(trace, varnames=varnames)
#     plt.tight_layout()
#     plt.show()

# # Plot posterior distributions for the coefficients
# plot_posterior(trace_4um, ['alpha', 'betas'])
# plot_posterior(trace_100nm, ['alpha', 'betas'])

# Plot posterior distributions for the coefficients using arviz
def plot_posterior(trace, varnames):
    az.plot_trace(trace, var_names=varnames)
    plt.tight_layout()
    plt.show()

# Plot posterior distributions for both datasets
plot_posterior(trace_4um, ['alpha', 'betas'])
plot_posterior(trace_100nm, ['alpha', 'betas'])

# Predict the spall strength using the posterior distributions
def predict_spall_strength(trace, X, y_err):
    alpha_post = trace['alpha']
    betas_post = trace['betas']
    
    predictions = alpha_post[:, None] + np.dot(X, betas_post.T)
    uncertainty = np.std(predictions, axis=1)  # Standard deviation as uncertainty
    
    return np.mean(predictions, axis=1), uncertainty

# Predict for 4 µm and 100 nm data
predictions_4um, uncertainty_4um = predict_spall_strength(trace_4um, X_4um, y_err_4um)
predictions_100nm, uncertainty_100nm = predict_spall_strength(trace_100nm, X_100nm, y_err_100nm)

# Plot predictions with uncertainty for both datasets
def plot_predictions_with_uncertainty(predictions, uncertainty, x_vals, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, predictions, label='Predicted Spall Strength')
    plt.errorbar(x_vals, predictions, yerr=uncertainty, fmt='o', color='r', label='Uncertainty')
    plt.title(title)
    plt.xlabel('Strain Rate (s^-1)')
    plt.ylabel('Predicted Spall Strength (GPa)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot for 4 µm and 100 nm
plot_predictions_with_uncertainty(predictions_4um, uncertainty_4um, data_4um['Strain Rate (s^-1)'], '4 µm - Predicted Spall Strength with Uncertainty')
plot_predictions_with_uncertainty(predictions_100nm, uncertainty_100nm, data_100nm['Strain Rate (s^-1)'], '100 nm - Predicted Spall Strength with Uncertainty')
# %%
