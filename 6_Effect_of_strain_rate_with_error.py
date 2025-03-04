# code with effect of strain rate including data uncertainity
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_all_csvs(file_pattern):
    """Plots data from all matching CSVs, both individually and combined, and saves the plots."""
    
    file_paths = glob.glob(file_pattern)
    if not file_paths:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    # Create a folder to save images
    output_folder = "f_strain_rate_with_error"
    os.makedirs(output_folder, exist_ok=True)
    
    # Lists to store all data for the combined plot
    all_spall_strength = []
    all_strain_rate = []
    all_strain_rate_err = []
    all_spall_strength_err = []
    
    for filepath in file_paths:
        try:
            df = pd.read_csv(filepath)
            try:
                spall_strength = df.loc[:, 'Spall Strength (GPa)']
                strain_rate = df.loc[:, 'Strain Rate (s^-1)']
                strain_rate_err = df.loc[:, 'Strain Rate Err (s^-1)']
                spall_strength_err = df.loc[:, 'Spall Strength Err (GPa)']
            except KeyError as e:
                print(f"Error: Required column not found in {filepath}: {e}")
                continue

            strain_rate_err = np.array(strain_rate_err)
            spall_strength_err = np.array(spall_strength_err)
            
            # Individual Plot
            plt.figure(figsize=(8, 6))
            plt.errorbar(strain_rate, spall_strength, xerr=strain_rate_err, yerr=spall_strength_err,
                         fmt='o', markersize=5, capsize=3, alpha=0.8, ecolor='gray', elinewidth=1.0, label='Data Points')
            plt.xlabel('Strain Rate (s^-1)')
            plt.ylabel('Spall Strength (GPa)')
            filename = os.path.basename(filepath)
            plt.title(f'Scatter Plot for {filename}')
            # plt.grid(True, linestyle='--', alpha=0.6)
            plt.xlim(5e5, 2e6)
            plt.ylim(1, 6)
            plt.legend()
            plt.tight_layout()
            
            # Save individual plot
            save_path = os.path.join(output_folder, f"{filename}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            # Append data for combined plot
            all_spall_strength.extend(spall_strength)
            all_strain_rate.extend(strain_rate)
            all_strain_rate_err.extend(strain_rate_err)
            all_spall_strength_err.extend(spall_strength_err)

        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Error processing {filepath}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with {filepath}: {e}")
    
    # Combined Plot (after processing all files)
    if all_spall_strength:
        all_spall_strength = np.array(all_spall_strength)
        all_strain_rate = np.array(all_strain_rate)
        all_strain_rate_err = np.array(all_strain_rate_err)
        all_spall_strength_err = np.array(all_spall_strength_err)

        plt.figure(figsize=(8, 6))
        plt.errorbar(all_strain_rate, all_spall_strength, xerr=all_strain_rate_err, yerr=all_spall_strength_err,
                     fmt='o', markersize=10, capsize=3, alpha=0.8, ecolor='gray', elinewidth=1.0, label='All Data')
        
        plt.xlabel('Strain Rate (s^-1)', fontsize=20)
        plt.ylabel('Spall Strength (GPa)', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xscale('log')
        plt.xlim(5e5, 2e6)
        plt.ylim(1, 6)
        plt.tight_layout()
        
        # Save combined plot
        combined_plot_path = os.path.join(output_folder, "combined_plot.png")
        plt.savefig(combined_plot_path, dpi=300)
        plt.close()
        
    else:
        print("No data was collected to create a combined plot.")

# Example usage:
file_pattern = '**_spall_strength_strain_rate_table_***mJ.csv'
plot_all_csvs(file_pattern)
