
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.signal import argrelextrema, butter, filtfilt
import os

# Constants
density_copper = 8950  # kg/m^3
acoustic_velocity_copper = 3950  # m/s
longitudanal_velocity_copper = 4750  # ms/s
hs = 100e-6  # flyer thickness in m

# Directory and file pattern 
input_directory = "/Users/piyushwanchoo/Library/CloudStorage/OneDrive-JohnsHopkins/Malon_PDV_scope/Combined_analysis/4um_800mJ_shots"  # Replace with your directory
file_pattern = "**--202*--*****--velocity.csv"  

files = sorted(glob.glob(os.path.join(input_directory, file_pattern)))
num_files = len(files)
print(f"Number of files in the folder: {num_files}")

# Initialize lists
first_maxima_list = []
first_maxima_err_list = []
minima_list = []
minima_err_list = []
spall_strength_list = []
spall_strength_err_list = []
strain_rate_list = []
strain_rate_err_list = []
velocity_after_minima_list = []
time_after_minima_list = []
recompression_slope_list = []
first_max_time_list = []
min_time_list = []
data_list = []

# Low-pass filter function
def low_pass_filter(data, cutoff_freq, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 24))

min_length = float('inf')

# 1. Process each file 
for file in files:
    data = pd.read_csv(file)
    X = data.iloc[:, 0]
    Y = data.iloc[:, 1]
    YErr = data.iloc[:, 2]

    dY_dX = np.gradient(Y, X)
    positive_slope_indices = np.where(dY_dX > 0)[0]

    if len(positive_slope_indices) > 0:
        idx = positive_slope_indices[np.argmin(np.abs(Y[positive_slope_indices] - 30))]
        X_shifted = (X - X[idx]) * 1e9 
    else:
        X_shifted = (X - X[0]) * 1e9

    # sampling_rate = 1 / (X.iloc[1] - X.iloc[0])

    if len(X) > 1 and (X.iloc[1] - X.iloc[0]) != 0:
        sampling_rate = 1 / (X.iloc[1] - X.iloc[0])
    else:
        print("Error: Insufficient or invalid time data.")
        exit()
    cutoff_freq = 10e7
    Y_filtered = low_pass_filter(Y.values, cutoff_freq, sampling_rate)

    Y = pd.Series(Y_filtered, index=Y.index)

    mask = X_shifted >= 0
    X_shifted_filtered = X_shifted[mask]
    Y_filtered = Y_filtered[mask]
    YErr_filtered = YErr[mask]

    min_length = min(min_length, len(X_shifted_filtered))
    data_list.append((X_shifted_filtered, Y_filtered, YErr_filtered))

    # data_list = [(x[:min_length], y[:min_length], z[:min_length]) for x, y, z in data_list]
    
    # Use the last dataset for further calculations
    X_trimmed, Y_trimmed, YErr_filtered = data_list[-1]  # Use the last processed dataset
    
    for X_trimmed, Y_trimmed, YErr_filtered in data_list:
        ax1.plot(X_trimmed, Y_trimmed, linestyle='-', linewidth=2)
        ax1.fill_between(X_trimmed, Y_trimmed - YErr_filtered, Y_trimmed + YErr_filtered, alpha=0.3)
    # (Maxima, minima, spall strength, strain rate calculations)
    max_time_range = (X_shifted >= 12) & (X_shifted <= 30)
    if np.any(max_time_range):
        Y_in_range_max = Y[max_time_range]
        X_in_range_max = X_shifted[max_time_range]

        # Find local maxima using argrelextrema
        local_maxima_idx = argrelextrema(Y_in_range_max.values, np.greater_equal, order=1)[0]

        if len(local_maxima_idx) > 0:
            first_max_idx = local_maxima_idx[0]
            max_value = Y_in_range_max.iloc[first_max_idx]
            max_time = X_in_range_max.iloc[first_max_idx]
            time_mask = np.isclose(X_trimmed, max_time, atol=1e-9)
            max_time_indices = np.where(time_mask)[0]

            # print(f"max_time: {max_time}")
            # print(f"X_trimmed shape: {X_trimmed.shape}")
            # print(f"YErr_filtered shape: {YErr_filtered.shape}")
            # print(f"max_time_indices: {max_time_indices}")

            if max_time_indices.size > 0:
                max_time_index = max_time_indices[0]
                max_value_err = YErr_filtered.iloc[max_time_index]
                first_maxima_list.append(max_value)
                first_maxima_err_list.append(max_value_err)
            else:
                max_value_err = np.nan
                first_maxima_list.append(max_value)
                first_maxima_err_list.append(max_value_err)
    time_range = (X_shifted >= 24) & (X_shifted <= 50)
    if np.any(time_range):
        Y_in_range_min = Y[time_range]
        X_in_range_min = X_shifted[time_range]
        local_minima_idx = argrelextrema(Y_in_range_min.values, np.less_equal, order=1)[0]

        if len(local_minima_idx) > 0:
            min_idx_after_max = local_minima_idx[0]
            min_value_after_max = Y_in_range_min.iloc[min_idx_after_max]
            min_time = X_in_range_min.iloc[min_idx_after_max]

            time_mask = np.isclose(X_trimmed, min_time, atol=1e-9)
            min_time_indices = np.where(time_mask)[0]

            if min_time_indices.size > 0:
                min_time_index = min_time_indices[0]
                min_value_err = YErr_filtered[min_time_index]
                minima_list.append(min_value_after_max)
                minima_err_list.append(min_value_err)
            else:
                min_value_err = np.nan
                minima_list.append(min_value_after_max)
                minima_err_list.append(min_value_err)

            # Spall Strength and Strain Rate Calculations (using uncertainties)
            spall_strength = (1 / 2) * density_copper * acoustic_velocity_copper * (max_value - min_value_after_max) * 1e-9

            spall_strength_err = np.sqrt((0.5 * density_copper * acoustic_velocity_copper * max_value_err * 1e-9)**2 + \
                                            (0.5 * density_copper * acoustic_velocity_copper * min_value_err * 1e-9)**2)

            strain_rate = -((max_value - min_value_after_max) / (2 * acoustic_velocity_copper * (max_time - min_time) * 1e-9))
            strain_rate_err = np.sqrt((1/(2*acoustic_velocity_copper*(max_time-min_time)*1e-9))**2 * (max_value_err**2 + min_value_err**2))

            # Append to lists 
            # first_maxima_list.append(max_value)
            # first_maxima_err_list.append(max_value_err)
            # minima_list.append(min_value_after_max)
            # minima_err_list.append(min_value_err)
            spall_strength_list.append(spall_strength)
            spall_strength_err_list.append(spall_strength_err)
            strain_rate_list.append(strain_rate)
            strain_rate_err_list.append(strain_rate_err)
            # first_max_time_list.append(max_time)
            # min_time_list.append(min_time)



            # Recompression peak detection and calculations
            recompression_range = (X_shifted >= 49) & (X_shifted <= 60)  # Example range
            if np.any(recompression_range):
                Y_in_range_recompression = Y[recompression_range]
                X_in_range_recompression = X_shifted[recompression_range]
                recompression_max_idx = argrelextrema(Y_in_range_recompression.values, np.greater_equal, order=1)[0]
                if len(recompression_max_idx) > 0:
                    recompression_idx = recompression_max_idx[0]
                    recompression_velocity = Y_in_range_recompression.iloc[recompression_idx]
                    recompression_time = X_in_range_recompression.iloc[recompression_idx]
                    velocity_after_minima_list.append(recompression_velocity)
                    time_after_minima_list.append(recompression_time)
                    recompression_slope = (recompression_velocity - min_value_after_max) / ((recompression_time * 1e-9) - (min_time * 1e-9))
                    recompression_slope_list.append(recompression_slope)

                    # Annotations (same as before)
                    ax1.annotate(f'Recompression Vel: {recompression_velocity:.2f}',
                                xy=(recompression_time, recompression_velocity),
                                xytext=(recompression_time, recompression_velocity + 50),
                                arrowprops=dict(facecolor='blue', shrink=0.05),
                                fontsize=12, color='blue')

            # Annotations for First Maxima and Minima (same as before)
            if len(local_maxima_idx) > 0:
                ax1.annotate(f'First Max: {max_value:.2f}',
                            xy=(max_time, max_value),
                            xytext=(max_time, max_value + 50),
                            arrowprops=dict(facecolor='green', shrink=0.05),
                            fontsize=12, color='green')

            # if len(local_minima_idx) > 0:
                ax1.annotate(f'Min: {min_value_after_max:.2f}',
                            xy=(X_in_range_min.iloc[min_idx_after_max], min_value_after_max),
                            xytext=(X_in_range_min.iloc[min_idx_after_max], min_value_after_max - 50),
                            arrowprops=dict(facecolor='red', shrink=0.05),
                            fontsize=12, color='red')


data_list = [(x[:min_length], y[:min_length], z[:min_length]) for x, y, z in data_list]

# --- 2. Calculate mean and std ---
max_length = max(len(x) for x, _, _ in data_list)
data_matrix = np.full((max_length, 2 * num_files), np.nan)

for i, (x, y, _) in enumerate(data_list):
    data_matrix[:len(x), 2 * i] = x
    data_matrix[:len(y), 2 * i + 1] = y

column_names = [f"{'X_shifted' if i % 2 == 0 else 'Y'}_{(i // 2) + 1}" for i in range(2 * num_files)]
df_matrix = pd.DataFrame(data_matrix, columns=column_names)

y_columns = df_matrix.iloc[:, 1::2]
mean_y = y_columns.mean(axis=1, skipna=True)
std_y = y_columns.std(axis=1, skipna=True)

x_first_column = df_matrix.iloc[:, 0].dropna().values
sample_frequency = x_first_column[2] - x_first_column[1]
# mean_x = np.arange(0, len(x_first_column) * sample_frequency, sample_frequency)
mean_x = df_matrix.iloc[:, 0].dropna().values
if len(mean_x) > len(x_first_column):
    mean_x = mean_x[:len(x_first_column)]

# 3. Plot mean and std 
ax2.errorbar(mean_x, mean_y, yerr=std_y, fmt='-', linewidth=2, color='black', ecolor='red', capsize=1)
ax2.set_xlabel('Time (ns)', fontsize=25)
ax2.set_ylabel('Mean Velocity (m/s)', fontsize=25)
ax2.set_title('Mean Velocity vs Time', fontsize=25)
ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='minor', labelsize=15)

# Save the mean velocity data (from the second plot) to a CSV file
mean_velocity_df = pd.DataFrame({
    'Time (ns)': mean_x,
    'Mean Velocity (m/s)': mean_y,
    'Error Bar (m/s)': std_y
})

## Save the DataFrame as a CSV file
mean_velocity_df.to_csv('4um_mean_velocity_vs_time_800mJ.csv', index=False)
print("Data from the second plot saved as 'mean_velocity_vs_time.csv'")

# Find first maximum on the mean plot
first_max_idx = np.argmax(mean_y)
Y_after_first_max = mean_y[first_max_idx:]

# Find minima that occurs after the first maximum
min_idx = np.argmin(Y_after_first_max)
min_idx = first_max_idx + min_idx
min_value = mean_y[min_idx]
min_time = mean_x[min_idx]

# Calculate the slope between first maxima and minima
slope = -(mean_y[first_max_idx] - min_value) / (mean_x[first_max_idx] - min_time)

# Calculate strain rate using the slope
strain_rate_mean = (slope) / (2 * acoustic_velocity_copper)


ax3.plot(mean_x, mean_y, label='Mean Velocity')
ax3.axvline(mean_x[first_max_idx], color='green', linestyle='--', label='First Max')
ax3.axvline(mean_x[min_idx], color='red', linestyle='--', label='Minima after First Max')
ax3.fill_betweenx([min_value, mean_y[first_max_idx]], mean_x[first_max_idx], mean_x[min_idx], color='gray', alpha=0.5)
ax3.set_xlabel('Time (ns)', fontsize=25)
ax3.set_ylabel('Mean Velocity (m/s)', fontsize=25)
ax3.set_title('Mean Velocity vs Time', fontsize=25)
# ax3.legend(loc='best')
ax3.grid(True)
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.tick_params(axis='both', which='minor', labelsize=15)

# 4. Final plot adjustments 
ax1.set_xlabel('Time (ns)', fontsize=25)
ax1.set_ylabel('Velocity (m/s)', fontsize=25)
ax1.set_title('Velocity vs Time', fontsize=25)
# ax1.legend(loc='best', fontsize=10, ncol=2)
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.tick_params(axis='both', which='minor', labelsize=15)
# ax1.set_xlim(left=0, right=0.06e3)  # Adjust the right limit as needed

# Set x and y-axis limits for all three plots
ax1.set_ylim(0, 700)
ax2.set_ylim(0, 700)
ax3.set_ylim(0, 700)
ax1.set_xlim(-10, 70)
ax2.set_xlim(-10, 70)
ax3.set_xlim(-10, 70)

plt.tight_layout()
plt.show()

   
# 5. DataFrame creation and saving 
# DataFrame creation using the lists populated in the first loop
max_length_df = max(
    len(first_maxima_list),
    len(first_maxima_err_list),
    len(minima_list),
    len(minima_err_list),
    len(velocity_after_minima_list),
    len(time_after_minima_list),
    len(spall_strength_list),
    len(spall_strength_err_list),
    len(strain_rate_list),
    len(strain_rate_err_list),
    len(recompression_slope_list)
)

def pad_list(lst, length):
    return lst + [np.nan] * (length - len(lst))

first_maxima_list = pad_list(first_maxima_list, max_length)
first_maxima_err_list = pad_list(first_maxima_err_list, max_length)
minima_list = pad_list(minima_list, max_length)
minima_err_list = pad_list(minima_err_list, max_length)
velocity_after_minima_list = pad_list(velocity_after_minima_list, max_length)
time_after_minima_list = pad_list(time_after_minima_list, max_length)
spall_strength_list = pad_list(spall_strength_list, max_length)
spall_strength_err_list = pad_list(spall_strength_err_list, max_length)
strain_rate_list = pad_list(strain_rate_list, max_length)
strain_rate_err_list = pad_list(strain_rate_err_list, max_length)
recompression_slope_list = pad_list(recompression_slope_list, max_length)

results_df = pd.DataFrame({
    'First Maxima (m/s)': first_maxima_list,
    'First Maxima Err (m/s)': first_maxima_err_list,
    'Minima (m/s)': minima_list,
    'Minima Err (m/s)': minima_err_list,
    'Recompression Velocity (m/s)': velocity_after_minima_list,
    'Time at Maxima after Minima (s)': [t * 1e-9 if t is not None else None for t in time_after_minima_list],  # Convert to seconds
    'Spall Strength (GPa)': spall_strength_list,
    'Spall Strength Err (GPa)': spall_strength_err_list, 
    'Strain Rate (s^-1)': strain_rate_list,
    'Strain Rate Err (s^-1)': strain_rate_err_list, 
    'Recompression Slope': recompression_slope_list
})

print(results_df)  # Print the DataFrame
results_df.to_csv('4um_spall_strength_strain_rate_table_800mJ.csv', index=False)  # Save to CSV

plt.show()   
# %%
