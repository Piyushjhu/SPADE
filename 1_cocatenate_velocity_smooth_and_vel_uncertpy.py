# %%
# this piece of code cocatinates velocity smooth and velocity uncertainity data together to
# create anew data file --velcoity.csv
import glob
import os
import pandas as pd

# Specify the directory path
directory = '/Users/piyushwanchoo/Library/CloudStorage/OneDrive-JohnsHopkins/Malon_PDV_scope/Combined_analysis/4um_1200mJ_shots_2'

# Get all files with the specified patterns
smooth_files = glob.glob(os.path.join(directory, '*--velocity--smooth.csv'))
uncert_files = glob.glob(os.path.join(directory, '*--vel--uncert.csv'))

# Create a dictionary to store file pairs
file_pairs = {}

# Group files with matching prefixes
for smooth_file in smooth_files:
    prefix = smooth_file.split('--velocity--smooth.csv')[0]
    file_pairs[prefix] = {'smooth': smooth_file}

for uncert_file in uncert_files:
    prefix = uncert_file.split('--vel--uncert.csv')[0]
    if prefix in file_pairs:
        file_pairs[prefix]['uncert'] = uncert_file

# Process each pair of files
for prefix, files in file_pairs.items():
    if 'smooth' in files and 'uncert' in files:
        # Read the smooth file
        df_smooth = pd.read_csv(files['smooth'])
        
        # Read the uncert file
        df_uncert = pd.read_csv(files['uncert'])
        
        # Create a new DataFrame with the desired columns
        df_combined = pd.DataFrame({
            'Column1': df_smooth.iloc[:, 0],
            'Column2': df_smooth.iloc[:, 1],
            'Column3': df_uncert.iloc[:, 1]
        })
        
        # Create the new filename
        new_filename = f"{prefix}--velocity.csv"
        
        # Save the combined DataFrame to a new CSV file
        df_combined.to_csv(os.path.join(directory, new_filename), index=False)
        
        print(f"Created: {new_filename}")

print("Processing complete.")

# %%
