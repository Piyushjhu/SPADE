

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Constants for Peak Shock Pressure calculation
density_copper = 8900  # kg/m^3
acoustic_velocity_copper = 3950  # m/s

# Load the CSV files
data_1000mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_1000mJ.csv")
data_1200mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_1200mJ.csv")
data_1600mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_1600mJ.csv")
data_800mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_800mJ.csv")
data_1350mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_1350mJ.csv")
data_1500mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_1500mJ.csv",
    skiprows=1,
    names=[
        "First Maxima (m/s)", "Minima (m/s)", "Recompression Velocity (m/s)",
        "Time at Maxima after Minima (s)", "Spall Strength (GPa)", "Strain Rate (s^-1)", "Recompression Slope"
    ]
)

# Drop header row in 1500mJ dataset and convert columns to numeric
data_1500mJ = data_1500mJ.drop(0).apply(pd.to_numeric, errors="coerce")

# Combine datasets for processing
datasets = {
    "1000mJ": data_1000mJ,
    "800mJ": data_800mJ,
    "1350mJ": data_1350mJ,
    "1500mJ": data_1500mJ
}

def remove_outliers_5sigma(data, x_col, y_col):
    mean_x, std_x = data[x_col].mean(), data[x_col].std()
    mean_y, std_y = data[y_col].mean(), data[y_col].std()
    return data[
        (np.abs(data[x_col] - mean_x) <= 1.2 * std_x) &
        (np.abs(data[y_col] - mean_y) <= 1.5 * std_y)
    ]

# Calculate Peak Shock Pressure for each dataset
for data in datasets.values():
    data['Peak Shock Pressure (GPa)'] = (
        data['First Maxima (m/s)'] * 0.5 * density_copper * acoustic_velocity_copper * 1e-9
    )

# Set up the first plot (scatter with translucent ellipses)
plt.figure(figsize=(15, 10))
cmap = cm.coolwarm
norm = mcolors.Normalize(vmin=4, vmax=8)  # Assuming a rough range for Peak Shock Pressure

for label, data in datasets.items():
    data_cleaned = remove_outliers_5sigma(data, "Peak Shock Pressure (GPa)", "Spall Strength (GPa)")
    x = data_cleaned["Peak Shock Pressure (GPa)"]
    y = data_cleaned["Spall Strength (GPa)"]
    
    # Scatter plot with colormap
    sc = plt.scatter(
        x, y, c=x, cmap=cmap, norm=norm, edgecolor="k", s=100, alpha=0.8, label=label
    )
    
    # Create translucent data cluster clouds
    if len(x) > 1 and len(y) > 1:
        cov = np.cov(x, y)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(
            xy=(np.mean(x), np.mean(y)),
            width=lambda_[0] * 2 * 2.5,
            height=lambda_[1] * 2 * 2.5,
            angle=np.rad2deg(np.arccos(v[0, 0])),
            color=cmap(norm(np.mean(x))),
            alpha=0.2
        )
        plt.gca().add_patch(ell)

# Add colorbar for first plot
cbar = plt.colorbar(sc)
cbar.set_label("Impact Velocity (m/s)", fontsize=20)
cbar.ax.tick_params(labelsize=1)

# Set labels and ticks for first plot
plt.xlabel("Peak Shock Pressure (GPa)", fontsize=25)
plt.ylabel("Spall Strength (GPa)", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0, 6])
plt.title("Spall Strength vs. Peak Shock Pressure", fontsize=25)

# Show the first plot
plt.tight_layout()
plt.show()

## 6 GPa is chosen as that is the minimum pressure at which spall is observed
# Set up the second plot (with trendline starting from 6 GPa)
plt.figure(figsize=(15, 10))

# Combine the data from 1000mJ, 1350mJ, and 1500mJ
combined_x = []
combined_y = []

for label, data in datasets.items():
    if label in ['1000mJ', '1350mJ', '1500mJ']:  # Only use these datasets for regression
        data_cleaned = remove_outliers_5sigma(data, "Peak Shock Pressure (GPa)", "Spall Strength (GPa)")
        x = data_cleaned["Peak Shock Pressure (GPa)"]
        y = data_cleaned["Spall Strength (GPa)"]

        # Append to combined lists
        combined_x.extend(x)
        combined_y.extend(y)

        # Scatter plot with colormap for second plot
        sc = plt.scatter(
            x, y, c=x, cmap=cmap, norm=norm, edgecolor="k", s=100, alpha=0.8, label=label
        )

# Fit a linear regression to the combined data (1000mJ, 1350mJ, 1500mJ)
combined_x = np.array(combined_x)
combined_y = np.array(combined_y)

# Perform linear fit (degree 1)
coeffs = np.polyfit(combined_x, combined_y, 1)  # Linear fit
trendline = np.poly1d(coeffs)

# Plot the trend line starting from 6 GPa on the x-axis
x_values = np.linspace(6, np.max(combined_x), 100)
y_values = trendline(x_values)
plt.plot(x_values, y_values, linestyle='--', color='black', linewidth=5, label='Trend Line (Combined Data)')

# Create translucent data cluster clouds for the second plot
for label, data in datasets.items():
    if label in ['1000mJ', '1350mJ', '1500mJ']:  # Only use these datasets for translucent clouds
        data_cleaned = remove_outliers_5sigma(data, "Peak Shock Pressure (GPa)", "Spall Strength (GPa)")
        x = data_cleaned["Peak Shock Pressure (GPa)"]
        y = data_cleaned["Spall Strength (GPa)"]
        if len(x) > 1 and len(y) > 1:
            cov = np.cov(x, y)
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(
                xy=(np.mean(x), np.mean(y)),
                width=lambda_[0] * 2 * 2.5,
                height=lambda_[1] * 2 * 2.5,
                angle=np.rad2deg(np.arccos(v[0, 0])),
                color=cmap(norm(np.mean(x))),
                alpha=0.2
            )
            plt.gca().add_patch(ell)

# Add colorbar for second plot
cbar = plt.colorbar(sc)
cbar.set_label("Impact Velocity (m/s)", fontsize=20)
cbar.ax.tick_params(labelsize=1)

# Set labels and ticks for second plot
plt.xlabel("Peak Shock Pressure (GPa)", fontsize=25)
plt.ylabel("Spall Strength (GPa)", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0, 6])
plt.title("Spall Strength vs. Peak Shock Pressure", fontsize=25)

# Show the second plot
plt.tight_layout()
plt.show()

# Get the data from 6 GPa onwards
mask = x_values >= 6  # Adjust this condition based on your x_values
x_filtered = x_values[mask]
y_filtered = y_values[mask]

# Perform linear regression using np.polyfit
slope, intercept = np.polyfit(x_filtered, y_filtered, 1)

# Print the slope
print(f"Slope of the regression line: {slope:.3f} GPa/GPa")
# %%
