# ElasticNet Regression
# For nanocrystalline data
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
output_dir = "ElasticNet_outputs_nanocrystalline"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
datasets = {
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

# Calculate Peak Shock Pressure
for dataset in datasets.values():
    dataset['Peak Shock Pressure (GPa)'] = dataset['First Maxima (m/s)'] * 0.5 * density_copper * acoustic_velocity_copper * 1e-9

# Combine datasets
all_data = pd.concat([
    datasets['800mJ'].assign(Energy='800mJ'),
    datasets['1000mJ'].assign(Energy='1000mJ'),
    datasets['1200mJ'].assign(Energy='1200mJ'),
    datasets['1350mJ'].assign(Energy='1350mJ'),
    datasets['1500mJ'].assign(Energy='1500mJ'),
    datasets['1600mJ'].assign(Energy='1600mJ')
]).reset_index(drop=True)

# Filter outliers
columns_to_check = ['Peak Shock Pressure (GPa)', 'Strain Rate (s^-1)', 'Spall Strength (GPa)']
all_data_filtered = remove_outliers(all_data, columns_to_check)

# Define features and target
X = all_data_filtered[['Peak Shock Pressure (GPa)', 'Strain Rate (s^-1)']]
y = all_data_filtered['Spall Strength (GPa)']

# Apply log transformation to strain rate
X['Log Strain Rate'] = np.log(X['Strain Rate (s^-1)'])

# Scale features
scaler = MinMaxScaler()
X['Scaled Strain Rate'] = scaler.fit_transform(X[['Strain Rate (s^-1)']])

# Keep Peak Shock Pressure unchanged
X_scaled = np.column_stack((X['Peak Shock Pressure (GPa)'].values, X['Scaled Strain Rate'].values))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train ElasticNet model with cross-validation
elasticnet_model = ElasticNetCV(cv=20, random_state=42, l1_ratio=0.5).fit(X_train, y_train)

# Predictions
y_train_pred = elasticnet_model.predict(X_train)
y_test_pred = elasticnet_model.predict(X_test)

# Metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics
print(f"Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"Testing MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

# Plot results
plt.figure(figsize=(14, 7))

# Training data plot
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, c='blue', alpha=0.6, label='Train Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Spall Strength (GPa)')
plt.ylabel('Predicted Spall Strength (GPa)')
plt.title('Training Data')
plt.legend()
plt.grid()

# Testing data plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, c='green', alpha=0.6, label='Test Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Spall Strength (GPa)')
plt.ylabel('Predicted Spall Strength (GPa)')
plt.title('Testing Data')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# Generate grid for surface plot
x_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
y_range = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Flatten the grid and create a DataFrame for prediction
grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]
Z_grid = elasticnet_model.predict(grid_points).reshape(X_grid.shape)

# Plot the surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(
    X_grid, Y_grid, Z_grid,
    cmap='viridis', alpha=0.8, edgecolor='none'
)

# Add data points
ax.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    y, color='red', label='Data Points', alpha=0.6
)

# Labels and title
ax.set_xlabel('Peak Shock Stress (GPa)')
ax.set_ylabel('Scaled Strain Rate (X 1e-6)')
ax.set_zlabel('Spall Strength (GPa)')
ax.set_title('Spall Strength Dependency on Strain Rate and Shock Pressure')

# Add a color bar for the surface
cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Spall Strength (GPa)')

plt.legend()
# plt.show()

# Create 3D rotating plot and save frames
gif_frames = []
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
for angle in range(0, 360, 5):
    ax.clear()
    ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], y, color='red', label='Data Points', alpha=0.6)
    ax.set_xlabel('Peak Shock Stress (GPa)')
    ax.set_ylabel('Scaled Strain Rate (X 1e-6)')
    ax.set_zlabel('Spall Strength (GPa)')
    ax.set_title('Rotating 3D Surface Plot')
    ax.view_init(30, angle)
    plt.savefig(f"{output_dir}/frame_{angle}.png")
    gif_frames.append(imageio.imread(f"{output_dir}/frame_{angle}.png"))

# Save GIF
imageio.mimsave(f"{output_dir}/rotating_surface.gif", gif_frames, duration=0.1)

print("All plots and GIF saved in 'ElasticNet_scaled_model' directory.")
# %%
