# Effect of strain rate with error individual energy comparisons

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

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
    output_folder = "f_strain_rate_with_error"
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

    # Individual and grouped plots
    for group, files in grouped_files.items():
        plt.figure(figsize=(8, 6))
        for idx, filepath in enumerate(files):
            try:
                df = pd.read_csv(filepath)
                spall_strength = df['Spall Strength (GPa)']
                strain_rate = df['Strain Rate (s^-1)']
                strain_rate_err = df['Strain Rate Err (s^-1)']
                spall_strength_err = df['Spall Strength Err (GPa)']

                legend_name = extract_legend_name(filepath)
                plt.errorbar(strain_rate, spall_strength, xerr=strain_rate_err, yerr=spall_strength_err,
                             fmt=markers[idx % len(markers)], color=colors[idx % len(colors)], markersize=8, capsize=3,
                             alpha=0.8, ecolor='gray', elinewidth=1.0, label=legend_name)

            except KeyError as e:
                print(f"Error: Missing column in {filepath}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error with {filepath}: {e}")
                continue

        plt.xlabel('Strain Rate (s^-1)', fontsize=20)
        plt.ylabel('Spall Strength (GPa)', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xscale('log')
        plt.xlim(5e5, 2e6)
        plt.ylim(1, 6)
        plt.legend(fontsize=14)
        plt.tight_layout()

        # Save grouped plot
        grouped_plot_path = os.path.join(output_folder, f"grouped_plot_{group}.png")
        plt.savefig(grouped_plot_path, dpi=300)
        plt.close()


file_pattern = '**_spall_strength_strain_rate_table_***mJ.csv'
plot_all_csvs(file_pattern)


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import glob
# import os
# from itertools import combinations

# def extract_suffix(filename, length=6):
#     """Extracts the last `length` characters (excluding extension) to use as a grouping key."""
#     base_name = os.path.basename(filename).rsplit('.', 1)[0]
#     return base_name[-length:]

# def plot_all_csvs(file_pattern):
#     file_paths = glob.glob(file_pattern)
#     if not file_paths:
#         print(f"No files found matching pattern: {file_pattern}")
#         return

#     output_folder = "f_strain_rate_with_error"
#     os.makedirs(output_folder, exist_ok=True)
    
#     grouped_files = {}
#     for filepath in file_paths:
#         suffix = extract_suffix(filepath)
#         grouped_files.setdefault(suffix, []).append(filepath)

#     # Generate combined plots for each group
#     for suffix, files in grouped_files.items():
#         if len(files) < 2:
#             continue  # Skip groups with less than two files
        
#         plt.figure(figsize=(8, 6))
#         colors = plt.cm.get_cmap("tab10", len(files))
        
#         for idx, filepath in enumerate(files):
#             df = pd.read_csv(filepath)
#             try:
#                 spall_strength = df['Spall Strength (GPa)']
#                 strain_rate = df['Strain Rate (s^-1)']
#                 strain_rate_err = df['Strain Rate Err (s^-1)']
#                 spall_strength_err = df['Spall Strength Err (GPa)']
#             except KeyError as e:
#                 print(f"Error: Required column not found in {filepath}: {e}")
#                 continue
            
#             plt.errorbar(strain_rate, spall_strength, xerr=strain_rate_err, yerr=spall_strength_err,
#                          fmt='o', markersize=5, capsize=3, alpha=0.8, ecolor='gray', 
#                          elinewidth=1.0, color=colors(idx), label=os.path.basename(filepath))
        
#         plt.xlabel('Strain Rate (s^-1)')
#         plt.ylabel('Spall Strength (GPa)')
#         plt.xscale('log')
#         plt.xlim(5e5, 2e6)
#         plt.ylim(1, 6)
#         plt.legend()
#         plt.tight_layout()
        
#         save_path = os.path.join(output_folder, f"grouped_plot_{suffix}.png")
#         plt.savefig(save_path, dpi=300)
#         plt.close()

#     print("Plots saved in", output_folder)

# # Example usage:
# file_pattern = '**_spall_strength_strain_rate_table_***mJ.csv'
# plot_all_csvs(file_pattern)

# %%
