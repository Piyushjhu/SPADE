# 7_spall_strength_f_shockstress_with_uncertainity_comparison
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Constants for shock stress calculation
density_copper = 8900  # kg/m³
acoustic_velocity_copper = 2907  # m/s

def extract_legend_name(filename):
    """Extracts the legend name using text before the first underscore and after the last underscore."""
    base = os.path.basename(filename).replace('.csv', '')
    parts = base.split('_')
    if len(parts) > 2:
        return f"{parts[0]}_{parts[-1]}"
    return base

def plot_all_csvs(file_pattern):
    """Plots data from all matching CSVs, both individually and grouped by suffix, and saves the plots."""

    file_paths = glob.glob(file_pattern)
    if not file_paths:
        print(f"No files found matching pattern: {file_pattern}")
        return

    # Create a folder to save images
    output_folder = "f_shock_stress_with_error"
    os.makedirs(output_folder, exist_ok=True)

    # Store files based on their last 6 characters before .csv
    grouped_files = {}
    for filepath in file_paths:
        base_name = os.path.basename(filepath)
        group_key = base_name[-10:-4]  # Extract last 6 characters before .csv
        grouped_files.setdefault(group_key, []).append(filepath)

    # Define markers and colors for different files
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'x']
    colors = ['b', 'r', 'g', 'c', 'm', 'orange', 'purple', 'brown']

    # Process and plot data
    for group, files in grouped_files.items():
        plt.figure(figsize=(8, 6))
        for idx, filepath in enumerate(files):
            try:
                df = pd.read_csv(filepath)

                # Calculate Peak Shock Pressure (GPa)
                df['Peak Shock Pressure (GPa)'] = (
                    df['First Maxima (m/s)'] * 0.5 * density_copper * acoustic_velocity_copper * 1e-9
                )

                # Calculate Shock Stress Error using First Maxima Error
                df['Shock Stress Err (GPa)'] = (
                    df['First Maxima Err (m/s)'] * 0.5 * density_copper * acoustic_velocity_copper * 1e-9
                )

                spall_strength = df['Spall Strength (GPa)']
                shock_stress = df['Peak Shock Pressure (GPa)']
                shock_stress_err = df['Shock Stress Err (GPa)']
                spall_strength_err = df['Spall Strength Err (GPa)']

                legend_name = extract_legend_name(filepath)
                plt.errorbar(shock_stress, spall_strength, xerr=shock_stress_err, yerr=spall_strength_err,
                             fmt=markers[idx % len(markers)], color=colors[idx % len(colors)], markersize=8, capsize=3,
                             alpha=0.8, ecolor='gray', elinewidth=1.0, label=legend_name)

            except KeyError as e:
                print(f"Error: Missing column in {filepath}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error with {filepath}: {e}")
                continue

        plt.xlabel('Peak Shock Pressure (GPa)', fontsize=20)
        plt.ylabel('Spall Strength (GPa)', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(3, 8)  
        plt.ylim(2, 5)
        plt.legend(fontsize=14)
        plt.tight_layout()

        # Save grouped plot
        grouped_plot_path = os.path.join(output_folder, f"grouped_plot_{group}.png")
        plt.savefig(grouped_plot_path, dpi=300)
        plt.close()

# Example usage
file_pattern = '**_spall_strength_strain_rate_table_***mJ.csv'
plot_all_csvs(file_pattern)
# %%
