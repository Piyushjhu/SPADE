# Combined Elastic net 
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import os
import imageio
from mpl_toolkits.mplot3d import Axes3D

plt.ion()

# Create output directory
output_dir = "ElasticNet_outputs_combined"
os.makedirs(output_dir, exist_ok=True)

# Load datasets for 4um grain size
datasets_4um = {
    '800mJ': pd.read_csv('4um_spall_strength_strain_rate_table_800mJ.csv'),
    '1200mJ': pd.read_csv('4um_spall_strength_strain_rate_table_1200mJ.csv'),
    '1350mJ': pd.read_csv('4um_spall_strength_strain_rate_table_1350mJ.csv'),
    '1500mJ': pd.read_csv('4um_spall_strength_strain_rate_table_1500mJ.csv')
}

# Load datasets for 100nm grain size
datasets_100nm = {
    '800mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_800mJ.csv'),
    '1000mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_1000mJ.csv'),
    '1200mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_1200mJ.csv'),
    '1350mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_1350mJ.csv'),
    '1500mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_1500mJ.csv'),
    '1600mJ': pd.read_csv('100nm_spall_strength_strain_rate_table_1600mJ.csv')
}



# Function to remove outliers
def remove_outliers(data, columns, z_threshold=3):
    z_scores = np.abs(stats.zscore(data[columns]))
    return data[(z_scores < z_threshold).all(axis=1)]

# Constants
density_copper = 8950  # kg/m^3
acoustic_velocity_copper = 3950  # m/s

# # Calculate Peak Shock Pressure for both datasets
# for dataset in {**datasets_4um, **datasets_100nm}.values():
#     dataset['Peak Shock Pressure (GPa)'] = dataset['First Maxima (m/s)'] * 0.5 * density_copper * acoustic_velocity_copper * 1e-9

# Calculate Peak Shock Pressure for 100nm dataset (already exists)
for dataset in datasets_100nm.values():
    dataset['Peak Shock Pressure (GPa)'] = dataset['First Maxima (m/s)'] * 0.5 * density_copper * acoustic_velocity_copper * 1e-9

# Calculate Peak Shock Pressure for 4um dataset (missing)
for dataset in datasets_4um.values():
    dataset['Peak Shock Pressure (GPa)'] = dataset['First Maxima (m/s)'] * 0.5 * density_copper * acoustic_velocity_copper * 1e-9


# Combine both datasets (4um and 100nm)
all_data_4um = pd.concat([
    datasets_4um['800mJ'].assign(Energy='800mJ'),
    datasets_4um['1200mJ'].assign(Energy='1200mJ'),
    datasets_4um['1350mJ'].assign(Energy='1350mJ'),
    datasets_4um['1500mJ'].assign(Energy='1500mJ')
]).reset_index(drop=True)

all_data_100nm = pd.concat([
    datasets_100nm['800mJ'].assign(Energy='800mJ'),
    datasets_100nm['1000mJ'].assign(Energy='1000mJ'),
    datasets_100nm['1200mJ'].assign(Energy='1200mJ'),
    datasets_100nm['1350mJ'].assign(Energy='1350mJ'),
    datasets_100nm['1500mJ'].assign(Energy='1500mJ'),
    datasets_100nm['1600mJ'].assign(Energy='1600mJ')
]).reset_index(drop=True)

print("Columns in 4um dataset:")
print(all_data_4um.columns)

print("\nColumns in 100nm dataset:")
print(all_data_100nm.columns)


# Filter outliers for both datasets
columns_to_check = ['Peak Shock Pressure (GPa)', 'Strain Rate (s^-1)', 'Spall Strength (GPa)']
all_data_4um_filtered = remove_outliers(all_data_4um, columns_to_check)
all_data_100nm_filtered = remove_outliers(all_data_100nm, columns_to_check)




# Define features and target for both datasets
X_4um = all_data_4um_filtered[['Peak Shock Pressure (GPa)', 'Strain Rate (s^-1)']]
y_4um = all_data_4um_filtered['Spall Strength (GPa)']

X_100nm = all_data_100nm_filtered[['Peak Shock Pressure (GPa)', 'Strain Rate (s^-1)']]
y_100nm = all_data_100nm_filtered['Spall Strength (GPa)']
# Check the column names in the datasets

# Apply log transformation to strain rate for both datasets
X_4um['Log Strain Rate'] = np.log(X_4um['Strain Rate (s^-1)'])
X_100nm['Log Strain Rate'] = np.log(X_100nm['Strain Rate (s^-1)'])

# Scale features for both datasets
scaler = MinMaxScaler()
X_4um['Scaled Strain Rate'] = scaler.fit_transform(X_4um[['Strain Rate (s^-1)']])
X_100nm['Scaled Strain Rate'] = scaler.fit_transform(X_100nm[['Strain Rate (s^-1)']])

# Keep Peak Shock Pressure unchanged
X_4um_scaled = np.column_stack((X_4um['Peak Shock Pressure (GPa)'].values, X_4um['Scaled Strain Rate'].values))
X_100nm_scaled = np.column_stack((X_100nm['Peak Shock Pressure (GPa)'].values, X_100nm['Scaled Strain Rate'].values))

# Train ElasticNet model for both datasets
X_train_4um, X_test_4um, y_train_4um, y_test_4um = train_test_split(X_4um_scaled, y_4um, test_size=0.3, random_state=42)
X_train_100nm, X_test_100nm, y_train_100nm, y_test_100nm = train_test_split(X_100nm_scaled, y_100nm, test_size=0.3, random_state=42)

# Train ElasticNet model for 4um dataset
elasticnet_4um = ElasticNetCV(cv=20, random_state=42).fit(X_train_4um, y_train_4um)

# Train ElasticNet model for 100nm dataset
elasticnet_100nm = ElasticNetCV(cv=20, random_state=42).fit(X_train_100nm, y_train_100nm)

# Generate grid for surface plot
x_range = np.linspace(X_4um_scaled[:, 0].min(), X_4um_scaled[:, 0].max(), 100)
y_range = np.linspace(X_4um_scaled[:, 1].min(), X_4um_scaled[:, 1].max(), 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Flatten the grid and create a DataFrame for prediction for both datasets
grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]
Z_grid_4um = elasticnet_4um.predict(grid_points).reshape(X_grid.shape)
Z_grid_100nm = elasticnet_100nm.predict(grid_points).reshape(X_grid.shape)

# Plot the surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 4um dataset
ax.plot_surface(X_grid, Y_grid, Z_grid_4um, cmap='viridis', alpha=0.6, edgecolor='none')
ax.scatter(X_4um_scaled[:, 0], X_4um_scaled[:, 1], y_4um, color='red', label='4µm Data Points', alpha=0.6)

# Plot 100nm dataset
ax.plot_surface(X_grid, Y_grid, Z_grid_100nm, cmap='plasma', alpha=0.6, edgecolor='none')
ax.scatter(X_100nm_scaled[:, 0], X_100nm_scaled[:, 1], y_100nm, color='blue', label='100nm Data Points', alpha=0.6)

# Labels and title
ax.set_xlabel('Peak Shock Stress (GPa)')
ax.set_ylabel('Scaled Strain Rate (X 1e-6)')
ax.set_zlabel('Spall Strength (GPa)')
ax.set_title('Spall Strength Dependency on Strain Rate and Shock Pressure')

# Add a color bar for the surface
cbar = fig.colorbar(ax.plot_surface(X_grid, Y_grid, Z_grid_4um, cmap='viridis', alpha=0.6, edgecolor='none'), ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Spall Strength (GPa)')

# Add legend
plt.legend()

# Create 3D rotating plot and save frames
gif_frames = []
for angle in range(0, 360, 5):
    ax.clear()
    ax.plot_surface(X_grid, Y_grid, Z_grid_4um, cmap='viridis', alpha=0.6, edgecolor='none')
    ax.plot_surface(X_grid, Y_grid, Z_grid_100nm, cmap='plasma', alpha=0.6, edgecolor='none')
    ax.scatter(X_4um_scaled[:, 0], X_4um_scaled[:, 1], y_4um, color='red', label='4µm Data Points', alpha=0.6)
    ax.scatter(X_100nm_scaled[:, 0], X_100nm_scaled[:, 1], y_100nm, color='blue', label='100nm Data Points', alpha=0.6)
    ax.set_xlabel('Peak Shock Stress (GPa)')
    ax.set_ylabel('Scaled Strain Rate (X 1e-6)')
    ax.set_zlabel('Spall Strength (GPa)')
    ax.set_title('Rotating 3D Surface Plot')
    ax.view_init(30, angle)
    plt.savefig(f"{output_dir}/frame_{angle}.png")
    gif_frames.append(imageio.imread(f"{output_dir}/frame_{angle}.png"))

# Save GIF
imageio.mimsave(f"{output_dir}/rotating_surface.gif", gif_frames, duration=0.1)

print("All plots and GIF saved in 'ElasticNet_outputs' directory.")
# %%
