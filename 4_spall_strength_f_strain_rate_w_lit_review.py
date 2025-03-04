
# # %%
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Try to read the CSV file without specifying column names
combined_lit_table = pd.read_csv("combined_lit_table.csv")

# Check the first few rows of the dataset to see its structure
print("Combined Literature Data (Raw):")
print(combined_lit_table.head())

#
combined_lit_table = pd.read_csv("combined_lit_table.csv", delimiter=',')  # Try a comma delimiter

# Check again after trying the correct delimiter
print("Combined Literature Data (After Delimiter Fix):")
print(combined_lit_table.head())


# Load the datasets
# Load the datasets
data_800mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_800mJ.csv")
data_1000mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_1000mJ.csv")
data_1200mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_1200mJ.csv")
data_1350mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_1350mJ.csv")
data_1600mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_1600mJ.csv")
data_1500mJ = pd.read_csv("100nm_spall_strength_strain_rate_table_1500mJ.csv", skiprows=1, names=[
    "First Maxima (m/s)", "Minima (m/s)", "Recompression Velocity (m/s)",
    "Time at Maxima after Minima (s)", "Spall Strength (GPa)", "Strain Rate (s^-1)", "Recompression Slope"
])
data_4um_1500mJ = pd.read_csv("4um_spall_strength_strain_rate_table_1500mJ.csv")
data_4um_1200mJ = pd.read_csv("4um_spall_strength_strain_rate_table_1200mJ.csv")
data_4um_800mJ = pd.read_csv("4um_spall_strength_strain_rate_table_800mJ.csv")
# data_SC100_1500mJ = pd.read_csv("SC_100_spall_strength_strain_rate_table_1500mJ.csv")
# data_SC110_1500mJ = pd.read_csv("SC_110_spall_strength_strain_rate_table_1500mJ.csv")
# data_SC111_1500mJ = pd.read_csv("SC_111_spall_strength_strain_rate_table_1500mJ.csv")
# data_1500mJ = data_1500mJ.drop(0).apply(pd.to_numeric, errors='coerce')  # Convert to numeric

# Combine the data for "This Study"
this_study_data = pd.concat([data_800mJ, data_1000mJ, data_1200mJ, data_1350mJ, data_1500mJ,data_1600mJ], ignore_index=True)
this_study_data2=pd.concat([data_4um_800mJ,data_4um_1200mJ, data_4um_1500mJ], ignore_index=True)
# this_study_data3=pd.concat([data_SC100_1500mJ, data_SC110_1500mJ, data_SC111_1500mJ], ignore_index=True)

# Load the literature data
combined_lit_table = pd.read_csv("combined_lit_table_only_poly.csv", header=None, names=[
    "Strain Rate (s^-1)", "Col2", "Col3", "Spall Strength (GPa)", "Source"
])

# Ensure all data is numeric where needed
this_study_data['Strain Rate (s^-1)'] = pd.to_numeric(this_study_data['Strain Rate (s^-1)'], errors='coerce')
this_study_data['Spall Strength (GPa)'] = pd.to_numeric(this_study_data['Spall Strength (GPa)'], errors='coerce')
this_study_data2['Strain Rate (s^-1)'] = pd.to_numeric(this_study_data2['Strain Rate (s^-1)'], errors='coerce')
this_study_data2['Spall Strength (GPa)'] = pd.to_numeric(this_study_data2['Spall Strength (GPa)'], errors='coerce')
# this_study_data3['Strain Rate (s^-1)'] = pd.to_numeric(this_study_data3['Strain Rate (s^-1)'], errors='coerce')
# this_study_data3['Spall Strength (GPa)'] = pd.to_numeric(this_study_data3['Spall Strength (GPa)'], errors='coerce')
combined_lit_table['Strain Rate (s^-1)'] = pd.to_numeric(combined_lit_table['Strain Rate (s^-1)'], errors='coerce')
combined_lit_table['Spall Strength (GPa)'] = pd.to_numeric(combined_lit_table['Spall Strength (GPa)'], errors='coerce')

# Drop rows with missing or invalid data
this_study_data = this_study_data.dropna(subset=['Strain Rate (s^-1)', 'Spall Strength (GPa)'])
this_study_data2 = this_study_data2.dropna(subset=['Strain Rate (s^-1)', 'Spall Strength (GPa)'])
# this_study_data3 = this_study_data3.dropna(subset=['Strain Rate (s^-1)', 'Spall Strength (GPa)'])
combined_lit_table = combined_lit_table.dropna(subset=['Strain Rate (s^-1)', 'Spall Strength (GPa)', 'Source'])

# Debugging: Print data summaries
print("This Study Data:", this_study_data.head(), sep="\n")
print("This Study Data2:", this_study_data2.head(), sep="\n")
# print("This Study Data3:", this_study_data3.head(), sep="\n")
print("Combined Literature Data:", combined_lit_table.head(), sep="\n")

# Plotting 1
plt.figure(figsize=(12, 10), dpi=300)


# Plot "This Study" data with a single color and symbol (e.g., 'o')
# plt.scatter(this_study_data['Strain Rate (s^-1)'], this_study_data['Spall Strength (GPa)'], 
#             s=150, alpha=0.6, color='cyan', label="This Study", marker='o')

# Plot literature data, grouped by Source
source_markers = ['o', 's', '^', 'D', 'v', '<', '>','*','d','p','h','H','+','x']  # Different marker styles for each source
colors = plt.cm.tab20(np.linspace(0, 1, combined_lit_table['Source'].nunique()))

for idx, source in enumerate(combined_lit_table['Source'].unique()):
    source_data = combined_lit_table[combined_lit_table['Source'] == source]
    plt.scatter(source_data['Strain Rate (s^-1)'], source_data['Spall Strength (GPa)'], 
                s=150, alpha=0.8, color=colors[idx], label=f"{source}", marker=source_markers[idx])

# Customize the plot
# plt.title('Literature Data', fontsize=25)
plt.xlabel('Strain Rate (s⁻¹)', fontsize=25)
plt.ylabel('Spall Strength (GPa)', fontsize=25)
plt.xscale('log')  # Use log scale for strain rate
plt.legend(fontsize=15, loc='upper left', frameon=True)
plt.tick_params(axis='both', which='major', labelsize=25)  # You can set labelsize for x and y axis
plt.tight_layout()
# Set axis limits
plt.xlim(1e3, 1e8)  # Example: x-axis limits from 1e3 to 1e7
plt.ylim(0, 8)      # Example: y-axis limits from 0 to 16 GPa
plt.show()


# Plotting 2
plt.figure(figsize=(12, 10), dpi=300)

# Plot "This Study" data with a single color and symbol (e.g., 'o')
plt.scatter(this_study_data2['Strain Rate (s^-1)'], this_study_data2['Spall Strength (GPa)'], 
            s=150, alpha=0.6, color='cyan', label="This Study_Poly", marker='o')

# Plot literature data, grouped by Source
source_markers = ['o', 's', '^', 'D', 'v', '<', '>','*','d','p','h','H','+','x']  # Different marker styles for each source
colors = plt.cm.tab20(np.linspace(0, 1, combined_lit_table['Source'].nunique()))

for idx, source in enumerate(combined_lit_table['Source'].unique()):
    source_data = combined_lit_table[combined_lit_table['Source'] == source]
    plt.scatter(source_data['Strain Rate (s^-1)'], source_data['Spall Strength (GPa)'], 
                s=150, alpha=0.1, color=colors[idx], label=f"{source}", marker=source_markers[idx])

# Customize the plot
# plt.title('Spall Strength vs. Strain Rate', fontsize=25)
plt.xlabel('Strain Rate (s⁻¹)', fontsize=25)
plt.ylabel('Spall Strength (GPa)', fontsize=25)
plt.xscale('log')  # Use log scale for strain rate
plt.legend(fontsize=15, loc='upper left', frameon=True)
plt.tick_params(axis='both', which='major', labelsize=25)  # You can set labelsize for x and y axis
plt.tight_layout()
# Set axis limits
plt.xlim(1e3, 1e8)  # Example: x-axis limits from 1e3 to 1e7
plt.ylim(0, 8)      # Example: y-axis limits from 0 to 2 GPa
plt.show()

plt.show()


# Plotting 3
plt.figure(figsize=(12, 10), dpi=300)

# Plot "This Study" data with a single color and symbol (e.g., 'o')
plt.scatter(this_study_data['Strain Rate (s^-1)'], this_study_data['Spall Strength (GPa)'], 
            s=150, alpha=0.6, color='red', label="This Study_nano", marker='o')
plt.scatter(this_study_data2['Strain Rate (s^-1)'], this_study_data2['Spall Strength (GPa)'], 
            s=150, alpha=0.6, color='cyan', label="This Study_Poly", marker='o')

# Plot literature data, grouped by Source
source_markers = ['o', 's', '^', 'D', 'v', '<', '>','*','d','p','h','H','+','x']  # Different marker styles for each source
colors = plt.cm.tab20(np.linspace(0, 1, combined_lit_table['Source'].nunique()))

for idx, source in enumerate(combined_lit_table['Source'].unique()):
    source_data = combined_lit_table[combined_lit_table['Source'] == source]
    plt.scatter(source_data['Strain Rate (s^-1)'], source_data['Spall Strength (GPa)'], 
                s=150, alpha=0.1, color=colors[idx], label=f"{source}", marker=source_markers[idx])

# Customize the plot
# plt.title('Spall Strength vs. Strain Rate', fontsize=25)
plt.xlabel('Strain Rate (s⁻¹)', fontsize=25)
plt.ylabel('Spall Strength (GPa)', fontsize=25)
plt.xscale('log')  # Use log scale for strain rate
plt.legend(fontsize=15, loc='upper left', frameon=True)
plt.tick_params(axis='both', which='major', labelsize=25)  # You can set labelsize for x and y axis
plt.tight_layout()
# Set axis limits
plt.xlim(1e3, 1e8)  # Example: x-axis limits from 1e3 to 1e7
plt.ylim(0, 8)      # Example: y-axis limits from 0 to 2 GPa
plt.show()

plt.show()

# Plotting 4
plt.figure(figsize=(12, 10), dpi=300)

# Plot "This Study" data with a single color and symbol (e.g., 'o')
plt.scatter(this_study_data['Strain Rate (s^-1)'], this_study_data['Spall Strength (GPa)'], 
            s=150, alpha=0.6, color='red', label="This Study_nano", marker='o')
plt.scatter(this_study_data2['Strain Rate (s^-1)'], this_study_data2['Spall Strength (GPa)'], 
            s=150, alpha=0.6, color='cyan', label="This Study_Poly", marker='o')
# plt.scatter(this_study_data3['Strain Rate (s^-1)'], this_study_data3['Spall Strength (GPa)'], 
            # s=150, alpha=0.6, color='black', label="This Study_single_crystals", marker='o')
# Plot literature data, grouped by Source
source_markers = ['o', 's', '^', 'D', 'v', '<', '>','*','d','p','h','H','+','x']  # Different marker styles for each source
colors = plt.cm.tab20(np.linspace(0, 1, combined_lit_table['Source'].nunique()))

for idx, source in enumerate(combined_lit_table['Source'].unique()):
    source_data = combined_lit_table[combined_lit_table['Source'] == source]
    plt.scatter(source_data['Strain Rate (s^-1)'], source_data['Spall Strength (GPa)'], 
                s=150, alpha=0.7, color=colors[idx], label=f"{source}", marker=source_markers[idx])

# Customize the plot
# plt.title('Spall Strength vs. Strain Rate', fontsize=25)
plt.xlabel('Strain Rate (s⁻¹)', fontsize=25)
plt.ylabel('Spall Strength (GPa)', fontsize=25)
plt.xscale('log')  # Use log scale for strain rate
plt.legend(fontsize=15, loc='upper left', frameon=True)
plt.tick_params(axis='both', which='major', labelsize=25)  # You can set labelsize for x and y axis
plt.tight_layout()
# Set axis limits
plt.xlim(1e3, 1e8)  # Example: x-axis limits from 1e3 to 1e7
plt.ylim(0, 8)      # Example: y-axis limits from 0 to 2 GPa
plt.show()

plt.show()

# %%
