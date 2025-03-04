# # # Older code with static plots
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Directory containing the input files
input_directory = "/Users/piyushwanchoo/Library/CloudStorage/OneDrive-JohnsHopkins/Malon_PDV_scope/Combined_analysis/Data_analysis_code_final_2"  
file_pattern = "100nm_mean_velocity_vs_time_*.csv"  
# Get a sorted list of all matching files
files = sorted(glob.glob(f"{input_directory}/{file_pattern}"))
print(f"Found {len(files)} files.")

# Define a dictionary to map filenames to legend labels
legend_mapping = {
    "0600mJ": "408 m/s",
    "0800mJ": "550 m/s",
    "1000mJ": "600 m/s",
    "1200mJ": "650 m/s",
    "1350mJ": "730 m/s",
    "1500mJ": "830 m/s",
    "1700mJ": "870 m/s",
}

# Create a figure for the plot
plt.figure(figsize=(12, 8))



# Function to create alpha gradient based on proximity to mean
def create_alpha_gradient(mean_velocity, velocity_std, max_alpha=0.2, min_alpha=0.02):
    extrema = np.maximum(np.abs(mean_velocity - velocity_std), np.abs(mean_velocity + velocity_std))
    relative_proximity = np.abs(mean_velocity - extrema) / extrema
    alpha_values = min_alpha + (max_alpha - min_alpha) * relative_proximity
    return alpha_values



# Loop through each file, process, and plot
for idx, file in enumerate(files):
    # Read the CSV file
    data = pd.read_csv(file)

    # Extract time, velocity, and standard deviation columns
    time = data.iloc[:, 0]  # Assuming the first column is time
    mean_velocity = data.iloc[:, 1]  # Assuming the second column is mean velocity
    velocity_std = data.iloc[:, 2]  # Assuming the third column is standard deviation

    # Create alpha gradient for error bars
    alpha_values = create_alpha_gradient(mean_velocity, velocity_std)

    # Extract the filename part needed for lookup
    filename_key = os.path.basename(file)[-10:-4]

    # Get the legend label from the dictionary, or use the filename if not found
    legend_label = legend_mapping.get(filename_key, filename_key)  # Improved lookup

    # Plot the mean velocity
    plt.plot(
        time,
        mean_velocity,
        linestyle='-',  # Solid line for mean
        linewidth=2,
        color=plt.cm.tab10(idx/len(files)), # More flexible color assignment
        label=legend_label  # Use the mapped label
    )

    # Plot standard deviation with transparency gradient
    for i in range(len(time) - 1):
        plt.fill_between(
            time[i:i + 2],
            (mean_velocity - velocity_std)[i:i + 2],
            (mean_velocity + velocity_std)[i:i + 2],
            color=plt.cm.tab10(idx/len(files)), # Consistent colors for line and fill
            alpha=alpha_values[i]
        )

# Customize the legend
plt.legend(fontsize=12, loc="best")

# Customize plot appearance
plt.xlabel("Time (ns)", fontsize=25)
plt.ylabel("Free Surface Velocity (m/s)", fontsize=25)
plt.title("Free surface velocity f(Impact velocity)", fontsize=25)
plt.ylim(0, 600)  # Adjusted y-axis limit
# plt.grid(True)
plt.tick_params(axis='both', labelsize=20)

# Show the plot
plt.tight_layout()
plt.show()
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import glob
# import os

# # Directory containing the input files
# input_directory = "/Users/piyushwanchoo/Library/CloudStorage/OneDrive-JohnsHopkins/Malon_PDV_scope/Cu_Polycrystals/Cu_100nm/Data_analysis_code_final"  # Replace with your directory
# file_pattern = "mean_velocity_vs_time_*.csv"  # Adjust this pattern to match your filenames

# # Get a sorted list of all matching files
# files = sorted(glob.glob(f"{input_directory}/{file_pattern}"))
# print(f"Found {len(files)} files.")

# # Define a list of colors for different files
# colors = plt.cm.tab10(np.linspace(0, 1, len(files)))  # Adjust colormap as needed

# # Extract last 6 characters (excluding '.csv') for legend labels
# legend_labels = [os.path.basename(file)[-10:-4] for file in files]  

# # Create a figure for the plot
# plt.figure(figsize=(12, 8))

# # Function to create alpha gradient based on proximity to mean
# def create_alpha_gradient(mean_velocity, velocity_std, max_alpha=0.2, min_alpha=0.02):
#     extrema = np.maximum(np.abs(mean_velocity - velocity_std), np.abs(mean_velocity + velocity_std))
#     relative_proximity = np.abs(mean_velocity - extrema) / extrema
#     alpha_values = min_alpha + (max_alpha - min_alpha) * relative_proximity
#     return alpha_values

# # Loop through each file, process, and plot
# for idx, file in enumerate(files):
#     # Read the CSV file
#     data = pd.read_csv(file)
    
#     # Extract time, velocity, and standard deviation columns
#     time = data.iloc[:, 0]  # Assuming the first column is time
#     mean_velocity = data.iloc[:, 1]  # Assuming the second column is mean velocity
#     velocity_std = data.iloc[:, 2]  # Assuming the third column is standard deviation

#     # Create alpha gradient for error bars
#     alpha_values = create_alpha_gradient(mean_velocity, velocity_std)

#     # Plot the mean velocity
#     plt.plot(
#         time,
#         mean_velocity,
#         linestyle='-',  # Solid line for mean
#         linewidth=2,
#         color=colors[idx],
#         label=legend_labels[idx]  # Use the last 6 characters of filename as label
#     )

#     # Plot standard deviation with transparency gradient
#     for i in range(len(time) - 1):
#         plt.fill_between(
#             time[i:i + 2],
#             (mean_velocity - velocity_std)[i:i + 2],
#             (mean_velocity + velocity_std)[i:i + 2],
#             color=colors[idx],
#             alpha=alpha_values[i]
#         )

# # Customize the legend
# plt.legend(fontsize=12, loc="best")

# # Customize plot appearance
# plt.xlabel("Time (ns)", fontsize=25)
# plt.ylabel("Free Surface Velocity (m/s)", fontsize=25)
# plt.title("Free surface velocity f(Impact velocity)", fontsize=25)
# plt.ylim(0, 600)
# # plt.grid(True)
# plt.tick_params(axis='both', labelsize=20)

# # Show the plot
# plt.tight_layout()
# plt.show()

# %%
