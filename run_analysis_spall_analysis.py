# %%
# --------------------------------------------------------------------------- #
# Example Analysis Script for Spall Analysis Package (Elastic Net Enabled)    #
# --------------------------------------------------------------------------- #
# Date: 2025-05-01                                                            #
# Description:                                                                #
# Runs the full analysis workflow, including per-material modeling (Step 8)   #
# and generates individual 3D plots per material, plus comparison plots.      #
# Includes an optional step (Step 9) for a combined 3D surface plot.          #
# Axes updated to match user request. Fixed data prep call in Step 8.         #
# Corrected plotting function calls in Step 3 based on log file errors.       #
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use('Agg') # Force non-interactive Agg backend
import spall_analysis as sa # Import the package with alias 'sa'
import os
import pandas as pd
import time # For timing the analysis
import matplotlib.pyplot as plt
import logging # Using logging as setup previously
import sys
import numpy as np # Needed for mean/std/interpolation
import glob # Needed for finding subfolders and tables
from scipy.interpolate import interp1d # For interpolating traces to common time grid
from datetime import datetime

# --- Logging Setup ---
log_filename = 'spall_analysis_run_elasticnet.log' # Use a different log file name
print(f"--- Detailed log will be saved to: {os.path.abspath(log_filename)} ---")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger().handlers[1].setLevel(logging.INFO) # Keep console output less verbose
logging.captureWarnings(True)
# --- End Logging Setup ---

# --- 1. Configuration ---
logging.info("--- Spall Analysis Workflow (Elastic Net Enabled) ---")
start_time = time.time()


BASE_DIR = '/Users/piyushwanchoo/Library/CloudStorage/OneDrive-JohnsHopkins/Malon_PDV_scope/Combined_analysis/Data_set'


OUTPUT_DIR = '/Users/piyushwanchoo/Desktop/spall_test_output_Final_2' 
logging.info(f"!!!! USING OUTPUT DIRECTORY: {OUTPUT_DIR} !!!!")


RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_velocity_traces')
LIT_DATA_DIR = os.path.join(BASE_DIR, 'literature_data')
LIT_FILE_STRAIN_RATE = os.path.join(LIT_DATA_DIR, 'combined_lit_table.csv')
# *** Use the same combined file, plotting function will check columns ***
LIT_FILE_SHOCK_STRESS = os.path.join(LIT_DATA_DIR, 'combined_lit_table.csv')

# --- Parameters for Data Processing (Dynamic Feature Detection + Plateau Mean) ---
raw_file_pattern_in_bin = '*.csv'
material_density = sa.DENSITY_COPPER
material_acoustic_velocity = sa.ACOUSTIC_VELOCITY_COPPER
# ** Parameters for dynamic feature detection - TUNE THESE **
processing_options = {
    'density': material_density,
    'acoustic_velocity': material_acoustic_velocity,
    'plot_individual': True,    # true- individual traces/ false- no individual traces
    'smooth_window': 7,         # Savitzky-Golay window
    'polyorder': 3,           # Savitzky-Golay polynomial order
    'prominence_factor': 0.05,  # Peak/valley prominence
    'peak_distance_ns': 5.0,    # Minimum time separation between features
    'plateau_duration_ns': 5.0  # Duration after peak to average for plateau velocity (Not used in Hybrid V3)
}

# --- Parameters for Plotting ---
# Options dictionary for Spall vs Strain Rate plot
plotting_options_strain = {
    'log_scale': True,
    'xlim': (1e4, 1e8), # Adjusted based on previous discussion for Expansion Rate
    'ylim': (0, 8),
    'filter_high_error_perc': 100, # Filter points with >100% relative error
    # 'x_col': 'Strain Rate (s^-1)', # REMOVED - Function expects 'strain_rate_col'
    'strain_rate_col': 'Strain Rate (s^-1)', # Expected argument
    'spall_col': 'Spall Strength (GPa)', # Expected argument
    'spall_unc_col': 'Spall Strength Err (GPa)', # Expected argument
    'material_col': 'Material', # Column to group by
    'group_by_material': True, # Group legend by material
    'show_legend': True # Added show_legend flag
}
# Options dictionary for Spall vs Shock Stress plot
plotting_options_shock = {
    'xlim': (0, 15), # Example range for shock stress
    'ylim': (0, 8),
    'filter_high_error_perc': 100,
    # 'shock_col': 'Peak Shock Pressure (GPa)', # REMOVED - Function expects 'shock_stress_col'
    'shock_stress_col': 'Peak Shock Pressure (GPa)', # Expected argument
    'spall_col': 'Spall Strength (GPa)', # Expected argument
    'spall_unc_col': 'Spall Strength Err (GPa)', # Expected argument
    'material_col': 'Material', # Column to group by
    'group_by_material': True, # Group legend by material
    'show_legend': True # Added show_legend flag
}

# --- Parameters for Complex Wilkerson Model ---
# Values updated based on Wilkerson & Ramesh, PRL 117, 215503 (2016)
wilkerson_params_base = {
    'sigma0_pa': 200e6,             # Pa
    'ky_sqrtm': 0.14e6,             # Pa*m^0.5
    'E_pa': 117e9,                  # Pa (Standard value, consistent with paper context)
    'Reos_pa': 22.5e9,              # Pa
    'K0_pa': 140e9,                 # Pa
    'rho': 8960.0,                  # kg/m^3
    'N2': 5000e18,                  # m^-3 (Grain interior sites) - Corrected exponent based on PRL units (um^-3 -> m^-3)
    'N0_GB': 10e18,                 # m^-3 (Reference GB site density) - Corrected exponent
    'd0_G': 100e-6,                 # m (Reference grain size for N0_GB)
}

# Define grain sizes and styles for multi-Wilkerson plot
grain_sizes_to_plot = {
    "Poly (4um)": 4.0e-6,
    "Nano (100nm)": 100.0e-9,
    "Single Crystal (Est. dG=1mm)": 1.0e-1 # Use large value or handle np.inf in model function
}
model_styles = {
    "Poly (4um)": "-",
    "Nano (100nm)": "--",
    "Single Crystal (Est. dG=1mm)": ":"
}
# Parameters for the single Wilkerson plot (example for 4um)
wilkerson_params_single = wilkerson_params_base.copy()
wilkerson_params_single['dG'] = 4.0e-6
# Options for the single Wilkerson plot
wilkerson_plot_options_single = {
    'xlim': (1e3, 1e9), # Adjusted for Expansion Rate
    'ylim': (0, 8),
    'spall_unc_col': 'Spall Strength Err (GPa)',
    'filter_high_error_perc': 100,
    'material_col': 'Material',
    'log_scale': True,
    'show_legend': True
}
# Options for the multi-Wilkerson plot
wilkerson_plot_options_multi = {
    'log_scale': True,
    'xlim': (1e3, 1e9), # Adjusted for Expansion Rate
    'ylim': (0, 8),
    'spall_unc_col': 'Spall Strength Err (GPa)',
    'filter_high_error_perc': 100,
    'material_col': 'Material',
    'group_by_material': True,
    'model_linestyle_map': model_styles,
    'show_legend': True # Added show_legend flag
}
# Options for combined mean trace plot
mean_trace_plot_options = {
    'xlim': (-5, 60), # Adjust time limits as needed
    'ylim': None, # Auto-scale Y
    'show_legend': True
    }

# --- Parameters for Modeling (Per-Material & Combined Plot) ---
model_features = [
    'Peak Shock Pressure (GPa)', # Use calculated shock pressure
    'Scaled Strain Rate'         # Use scaled strain rate
]
model_target = 'Spall Strength (GPa)'
plot_feature_x_3d = 'Peak Shock Pressure (GPa)' # Feature for 3D plot X-axis
plot_feature_y_3d = 'Scaled Strain Rate'        # Feature for 3D plot Y-axis
models_to_run = ['ElasticNet', 'Lasso']
model_transformations = {
    'Strain Rate (s^-1)': {'func': lambda x: x * 1e-6, 'out_name': 'Scaled Strain Rate'}
}
required_base_cols_for_modeling = [
    'First Maxima (m/s)',
    'Strain Rate (s^-1)',
    'Spall Strength (GPa)',
    'Material',
]
metadata_cols_to_keep_for_modeling = [
    'Material',
    'Strain Rate (s^-1)',
    'First Maxima (m/s)',
    # Add other original columns if needed
]
# Ensure model features and plotting features are kept
metadata_cols_to_keep_for_modeling = list(set(metadata_cols_to_keep_for_modeling + model_features + [plot_feature_x_3d, plot_feature_y_3d]))


# --- Create Output Directories ---
logging.info(f"Main output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
plot_output_dir = os.path.join(OUTPUT_DIR, 'plots')
table_output_dir = os.path.join(OUTPUT_DIR, 'tables')
model_output_dir = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(plot_output_dir, exist_ok=True)
os.makedirs(table_output_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)


# --- 2. Discover and Process Bins (Subfolders) ---
logging.info("\n--- Step 1: Discovering and Processing Data Bins ---")
try:
    subfolders = sorted([f.path for f in os.scandir(RAW_DATA_DIR) if f.is_dir()])
except FileNotFoundError:
    logging.error(f"Raw data directory not found: {RAW_DATA_DIR}. Please check BASE_DIR and RAW_DATA_DIR paths.")
    sys.exit()

if not subfolders:
    logging.error(f"No subfolders (data bins) found directly within {RAW_DATA_DIR}. Check directory structure.")
    sys.exit()

logging.info(f"Found {len(subfolders)} potential data bins (subfolders).")

all_bin_summary_files = []
all_mean_raw_files = []
all_bin_dfs = []

for subfolder_path in subfolders:
    subfolder_name = os.path.basename(subfolder_path)
    logging.info(f"\n Processing Bin: {subfolder_name} ".center(60, '-'))

    # --- A. Process traces and save bin-specific results table ---
    bin_summary_table_name = f"{subfolder_name}_results_table_dynamic_plateau.csv"
    bin_summary_table_path = os.path.join(table_output_dir, bin_summary_table_name)

    bin_summary_df = sa.process_velocity_files(
        input_folder=subfolder_path,
        file_pattern=raw_file_pattern_in_bin,
        output_folder=plot_output_dir,
        save_summary_table=True,
        summary_table_name=bin_summary_table_name,
        **processing_options
    )

    if bin_summary_df is None or bin_summary_df.empty:
        logging.warning(f"Processing bin '{subfolder_name}' produced no results or failed. Skipping mean trace calculations for this bin.")
        continue

    all_bin_summary_files.append(bin_summary_table_path)
    all_bin_dfs.append(bin_summary_df)
    logging.info(f"Finished processing bin '{subfolder_name}'. Results saved to '{bin_summary_table_path}'")

    successful_bin_df = bin_summary_df[bin_summary_df['Processing Status'] == 'Success'].copy()
    if successful_bin_df.empty:
        logging.warning(f"No traces processed successfully in bin '{subfolder_name}'. Skipping mean raw trace calculation.")
        continue

    # --- B. Calculate and save Mean Raw Velocity Trace for this bin ---
    logging.info(f"  Calculating mean raw velocity trace for bin '{subfolder_name}'...")
    bin_raw_files = sorted(glob.glob(os.path.join(subfolder_path, raw_file_pattern_in_bin)))
    if not bin_raw_files: logging.warning(f"  No raw files found in {subfolder_path} for mean calculation.")
    else:
        all_times = []; all_velocities = []; min_len = float('inf'); valid_trace_count = 0
        successful_raw_files = []
        for fname_no_ext in successful_bin_df['Filename']:
             found = False; possible_fnames = [fname_no_ext + '.csv', fname_no_ext]
             for fname_try in possible_fnames:
                 fpath = os.path.join(subfolder_path, fname_try)
                 if os.path.exists(fpath): successful_raw_files.append(fpath); found = True; break
             if not found: logging.warning(f" Raw file for {fname_no_ext} not found in {subfolder_path}. Skipping for mean.")
        logging.info(f"    Found {len(successful_raw_files)} corresponding raw files for mean calculation.")
        for raw_file in successful_raw_files:
             try:
                data = pd.read_csv(raw_file);
                if data.shape[1] < 2: logging.warning(f"    Skipping {os.path.basename(raw_file)} for mean calc: Needs >= 2 columns."); continue
                X_orig = data.iloc[:, 0]; Y_orig = data.iloc[:, 1]; X = pd.to_numeric(X_orig, errors='coerce'); Y = pd.to_numeric(Y_orig, errors='coerce'); valid_idx = X.notna() & Y.notna();
                if not valid_idx.any(): continue
                X = X[valid_idx]; Y = Y[valid_idx];
                if not X.is_monotonic_increasing: sort_idx = X.argsort(); X = X.iloc[sort_idx]; Y = Y.iloc[sort_idx]
                X = X.reset_index(drop=True); Y = Y.reset_index(drop=True);
                if len(X) < 2: continue
                x_vals_for_grad = X.values; y_vals_for_grad = Y.values;
                if len(x_vals_for_grad) < 2 : continue
                with np.errstate(divide='ignore', invalid='ignore'): dY_dX = np.gradient(y_vals_for_grad, x_vals_for_grad)
                positive_slope_indices = np.where(dY_dX > 0)[0]; idx_shift = 0
                if len(positive_slope_indices)>0:
                    try: candidate_values = Y.iloc[positive_slope_indices]; closest_label = np.abs(candidate_values - 30).idxmin(); idx_shift = Y.index.get_loc(closest_label)
                    except Exception: idx_shift = 0
                if not (0 <= idx_shift < len(X)): idx_shift = 0
                t_shift = X.iloc[idx_shift]; X_shifted = (X - t_shift) * 1e9; mask = X_shifted >= 0;
                if not mask.any(): continue
                X_shifted_filtered = X_shifted[mask].reset_index(drop=True); Y_filtered = Y[mask].reset_index(drop=True);
                if len(X_shifted_filtered) < 2: continue
                all_times.append(X_shifted_filtered); all_velocities.append(Y_filtered); min_len = min(min_len, len(X_shifted_filtered)); valid_trace_count += 1
             except Exception as e: logging.warning(f"    Could not read or preprocess raw file {os.path.basename(raw_file)} for mean calc: {e}")
        if valid_trace_count > 1 and min_len != float('inf') and min_len > 1:
            shortest_trace_idx = np.argmin([len(t) for t in all_times]); common_time_grid_raw = all_times[shortest_trace_idx].values[:min_len]; aligned_velocities = np.full((min_len, valid_trace_count), np.nan); current_valid_idx = 0
            for t, v in zip(all_times, all_velocities):
                try:
                    unique_mask = np.concatenate(([True], np.diff(t.values) > 1e-9));
                    if unique_mask.sum() < 2: logging.warning(f"    Skipping trace in {subfolder_name} for mean calc: Not enough unique time points after filtering."); continue
                    interp_func = interp1d(t.values[unique_mask], v.values[unique_mask], kind='linear', bounds_error=False, fill_value=np.nan); aligned_velocities[:, current_valid_idx] = interp_func(common_time_grid_raw); current_valid_idx += 1
                except Exception as e: logging.warning(f"    Interpolation failed for one trace in {subfolder_name}: {e}")
            if current_valid_idx > 1:
                aligned_velocities = aligned_velocities[:, :current_valid_idx]; mean_raw_vel = np.nanmean(aligned_velocities, axis=1); std_raw_vel = np.nanstd(aligned_velocities, axis=1); mean_raw_df = pd.DataFrame({'Time (ns)': common_time_grid_raw, 'Mean Velocity (m/s)': mean_raw_vel, 'Std Dev Velocity (m/s)': std_raw_vel}); mean_raw_filename = os.path.join(table_output_dir, f"{subfolder_name}_mean_raw_velocity.csv");
                try: mean_raw_df.to_csv(mean_raw_filename, index=False, float_format='%.4f'); all_mean_raw_files.append(mean_raw_filename); logging.info(f"    Saved mean raw velocity trace to {mean_raw_filename}")
                except Exception as e: logging.error(f"    Could not save mean raw velocity trace for {subfolder_name}: {e}")
            else: logging.warning(f"  Not enough valid traces ({current_valid_idx}) after interpolation for mean raw velocity calculation in {subfolder_name}.")
        elif valid_trace_count <= 1: logging.warning(f"  Need more than 1 valid trace to calculate mean raw velocity for {subfolder_name}.")
        else: logging.warning(f"  Could not determine common length/grid for mean raw velocity calculation in {subfolder_name}.")


# --- 4. Combine Results ---
logging.info("\n--- Step 2: Combining Results from All Bins ---")
combined_summary_df = None
if not all_bin_dfs:
    logging.error("No individual summary DataFrames were generated. Cannot combine results or create plots.")
    sys.exit()
try:
    combined_summary_df = pd.concat(all_bin_dfs, ignore_index=True)
    logging.info(f"Combined results from {len(all_bin_dfs)} bins into a single DataFrame with {len(combined_summary_df)} total entries.")
    combined_summary_table_name = "COMBINED_summary_results_dynamic_plateau.csv"
    combined_summary_table_path = os.path.join(table_output_dir, combined_summary_table_name)
    df_to_save = combined_summary_df.copy()
    for col in ['model_lines_info', 'model_intersections']:
         if col in df_to_save.columns: df_to_save[col] = df_to_save[col].apply(lambda x: str(x) if x is not None else '')
    df_to_save.to_csv(combined_summary_table_path, index=False, float_format='%.4e')
    logging.info(f"Combined summary table saved to: {combined_summary_table_path}")
except Exception as e:
    logging.exception(f"Error combining summary tables: {e}")
    sys.exit()


# --- 5. Generate Comparison Plots (Using Combined Data) ---
logging.info("\n--- Step 3: Generating Comparison Plots (Combined Data) ---")
processing_suffix = "dynamic_plateau" # Suffix for filenames
plot_spall_strain_path = os.path.join(plot_output_dir, f'COMBINED_spall_vs_strain_rate_{processing_suffix}.png')
plot_spall_shock_path = os.path.join(plot_output_dir, f'COMBINED_spall_vs_shock_stress_{processing_suffix}.png')

if combined_summary_df is not None:
    plot_df_exp_filtered = combined_summary_df[combined_summary_df['Processing Status'] == 'Success'].copy()
    if plot_df_exp_filtered.empty: logging.warning("No successfully processed experimental data found in combined results to plot.")
    else: logging.info(f"Using {len(plot_df_exp_filtered)} successfully processed traces from combined data for plots.")

    # --- Plot: Spall Strength vs. Strain Rate ---
    if plotting_options_strain['strain_rate_col'] in plot_df_exp_filtered.columns and plotting_options_strain['spall_col'] in plot_df_exp_filtered.columns:
        try:
            plot_args_strain = plotting_options_strain.copy() # Use the dictionary defined earlier
            plot_args_strain['df'] = plot_df_exp_filtered # Pass the filtered data
            plot_args_strain['output_filename'] = plot_spall_strain_path
            plot_args_strain['title'] = f'Combined Spall Strength vs. Strain Rate ({processing_suffix.replace("_", " ").title()})'
            plot_args_strain['literature_data_file'] = LIT_FILE_STRAIN_RATE
            plot_args_strain['color_map'] = sa.COLOR_MAPPING
            plot_args_strain['marker_map'] = sa.MARKER_MAPPING
            # Remove 'x_col' as it's not expected by the function anymore
            if 'x_col' in plot_args_strain: del plot_args_strain['x_col']

            logging.info(" Generating Combined Spall vs Strain Rate plot...")
            sa.plot_spall_vs_strain_rate(**plot_args_strain)
        except Exception as e: logging.exception(f"  Error generating Combined Spall vs Strain Rate plot: {e}")
    else: logging.warning(f"Skipping Combined Spall vs Strain Rate plot - Required columns missing.")

    # --- Calculate Shock Stress & Plot Spall vs Shock Stress ---
    vel_col_for_shock = 'First Maxima (m/s)'
    if vel_col_for_shock in plot_df_exp_filtered.columns:
        logging.info(f" Calculating Shock Stress using '{vel_col_for_shock}' for combined plot...")
        plot_df_exp_with_shock = sa.add_shock_stress_column(
            plot_df_exp_filtered.copy(),
            velocity_col=vel_col_for_shock,
            density=material_density,
            acoustic_velocity=material_acoustic_velocity,
            new_col_name='Peak Shock Pressure (GPa)'
        )
        logging.info(f" Generating Combined Spall vs Shock Stress plot...")
        shock_col_to_plot = 'Peak Shock Pressure (GPa)'

        if shock_col_to_plot in plot_df_exp_with_shock.columns and plotting_options_shock['spall_col'] in plot_df_exp_with_shock.columns:
            try:
                plot_args_shock = plotting_options_shock.copy()
                plot_args_shock['df'] = plot_df_exp_with_shock
                plot_args_shock['output_filename'] = plot_spall_shock_path
                plot_args_shock['title'] = f'Combined Spall Strength vs. Shock Stress ({processing_suffix.replace("_", " ").title()})'
                plot_args_shock['literature_data_file'] = LIT_FILE_SHOCK_STRESS
                plot_args_shock['lit_shock_col'] = 'Shock Stress (GPa)' # Expected lit column
                plot_args_shock['lit_spall_col'] = 'Spall Strength (GPa)' # Expected lit column
                plot_args_shock['lit_source_col'] = 'Source' # Expected lit column
                plot_args_shock['color_map'] = sa.COLOR_MAPPING
                plot_args_shock['marker_map'] = sa.MARKER_MAPPING
                # Remove 'shock_col' as it's not expected by the function anymore
                if 'shock_col' in plot_args_shock: del plot_args_shock['shock_col']

                sa.plot_spall_vs_shock_stress(**plot_args_shock)
            except Exception as e: logging.exception(f"  Error generating Combined Spall vs Shock Stress plot: {e}")
        else: logging.warning(f"Skipping Combined Spall vs Shock Stress plot - Required columns ('{shock_col_to_plot}', '{plotting_options_shock['spall_col']}') missing.")
    else: logging.warning(f"Skipping Combined Spall vs Shock Stress plot - Velocity column '{vel_col_for_shock}' not found.")
else:
    logging.error("Skipping combined plotting because combined summary DataFrame could not be created.")


# --- 6. Generate Wilkerson Comparison Plots (Using Combined Data) ---
logging.info("\n--- Step 4: Generating Wilkerson Comparison Plots (Combined Data) ---")
plot_wilkerson_single_path = os.path.join(plot_output_dir, f'COMBINED_spall_vs_strain_rate_wilkerson_single_{processing_suffix}.png')
plot_wilkerson_multi_path = os.path.join(plot_output_dir, f'COMBINED_spall_vs_strain_rate_multi_wilkerson_{processing_suffix}.png')

if 'plot_df_exp_filtered' in locals() and not plot_df_exp_filtered.empty:
    # --- Generate SINGLE Wilkerson Comparison Plot ---
    logging.info(f" Generating Combined Single Wilkerson comparison plot...")
    if plotting_options_strain['strain_rate_col'] in plot_df_exp_filtered.columns and plotting_options_strain['spall_col'] in plot_df_exp_filtered.columns:
        try:
            wilkerson_plot_args_single = wilkerson_plot_options_single.copy()
            wilkerson_plot_args_single['experimental_data_df'] = plot_df_exp_filtered
            wilkerson_plot_args_single['output_filename'] = plot_wilkerson_single_path
            wilkerson_plot_args_single['wilkerson_params'] = wilkerson_params_single
            wilkerson_plot_args_single['strain_rate_col'] = plotting_options_strain['strain_rate_col'] # Use correct key
            wilkerson_plot_args_single['spall_col'] = plotting_options_strain['spall_col']
            wilkerson_plot_args_single['color_map'] = sa.COLOR_MAPPING
            wilkerson_plot_args_single['marker_map'] = sa.MARKER_MAPPING
            # Ensure xlabel is set correctly for Expansion Rate
            wilkerson_plot_args_single['xlabel'] = 'Expansion Rate ($\dot{v}/v_0$) [s$^{-1}$]'
            sa.plot_wilkerson_comparison(**wilkerson_plot_args_single)
        except Exception as e: logging.exception(f"  Error generating Combined Single Wilkerson plot: {e}")
    else: logging.warning("Skipping Combined Single Wilkerson plot - Required columns missing.")

    # --- Generate MULTI-Wilkerson Comparison Plot ---
    logging.info(f" Generating Combined Multi-Wilkerson comparison plot...")
    if plotting_options_strain['strain_rate_col'] in plot_df_exp_filtered.columns and plotting_options_strain['spall_col'] in plot_df_exp_filtered.columns:
        try:
            multi_wilkerson_args = wilkerson_plot_options_multi.copy()
            multi_wilkerson_args['experimental_data_df'] = plot_df_exp_filtered
            multi_wilkerson_args['output_filename'] = plot_wilkerson_multi_path
            multi_wilkerson_args['literature_data_file'] = LIT_FILE_STRAIN_RATE
            multi_wilkerson_args['wilkerson_params_base'] = wilkerson_params_base
            multi_wilkerson_args['grain_sizes_dict'] = grain_sizes_to_plot
            multi_wilkerson_args['strain_rate_col'] = plotting_options_strain['strain_rate_col'] # Use correct key
            multi_wilkerson_args['spall_col'] = plotting_options_strain['spall_col']
            multi_wilkerson_args['color_map'] = sa.COLOR_MAPPING
            multi_wilkerson_args['marker_map'] = sa.MARKER_MAPPING
            # Ensure xlabel is set correctly for Expansion Rate
            multi_wilkerson_args['xlabel'] = 'Expansion Rate ($\dot{v}/v_0$) [s$^{-1}$]'
            sa.plot_spall_vs_strain_rate_multi_wilkerson(**multi_wilkerson_args)
        except Exception as e: logging.exception(f"  Error generating Combined Multi-Wilkerson plot: {e}")
    else: logging.warning("Skipping Combined Multi-Wilkerson plot - Required columns missing.")
else:
    logging.warning("Skipping Wilkerson plots as no successfully processed experimental data is available.")


# --- 7. Generate Combined Mean Raw Trace Plot ---
logging.info("\n--- Step 7: Generating Combined Mean Raw Trace Plot ---")
plot_combined_raw_path = os.path.join(plot_output_dir, f'COMBINED_mean_raw_velocity_traces_{processing_suffix}.png')
mean_raw_glob_pattern = os.path.join(table_output_dir, '*_mean_raw_velocity.csv')
logging.info(f"Looking for mean raw trace files matching: {mean_raw_glob_pattern}")
actual_mean_raw_files = sorted(glob.glob(mean_raw_glob_pattern))

if actual_mean_raw_files:
    logging.info(f"Found {len(actual_mean_raw_files)} mean raw trace files.")
    try:
        # Use the options defined earlier
        combined_trace_args = mean_trace_plot_options.copy()
        combined_trace_args['mean_trace_files'] = actual_mean_raw_files
        combined_trace_args['output_filename'] = plot_combined_raw_path
        combined_trace_args['material_map'] = sa.MATERIAL_MAPPING
        combined_trace_args['color_map'] = sa.COLOR_MAPPING
        combined_trace_args['legend_mapping'] = sa.ENERGY_VELOCITY_MAPPING

        sa.plot_combined_mean_traces(**combined_trace_args)
    except Exception as e:
        logging.exception(f" Error generating combined mean raw traces plot: {e}")
else:
    logging.warning(f"Skipping combined mean raw traces plot - no mean raw trace files found matching pattern in {table_output_dir}.")


# --- Step 8: Per-Material Modeling and 3D Plots (Optional, uncomment to run) ---
# logging.info("\n--- Step 8: Training Models and Generating 3D Plots Per Material ---")
# if 'combined_summary_df' in locals() and combined_summary_df is not None and not combined_summary_df.empty:
#     model_input_df = combined_summary_df[combined_summary_df['Processing Status'] == 'Success'].copy()
#     if not model_input_df.empty:
#         logging.info(f"Preparing data for per-material modeling using {len(model_input_df)} successful traces...")
#         X_full, y_full, df_full_prepared = sa.prepare_feature_matrix(
#             data_dict={'combined_successful': model_input_df},
#             feature_cols=model_features,
#             target_col=model_target,
#             required_cols=required_base_cols_for_modeling,
#             metadata_cols_to_keep=metadata_cols_to_keep_for_modeling,
#             calculate_shock_stress_if_missing=True,
#             velocity_col='First Maxima (m/s)',
#             density=material_density,
#             acoustic_velocity=material_acoustic_velocity,
#             transformations=model_transformations
#         )
#         if X_full is not None and y_full is not None and df_full_prepared is not None:
#             logging.info("Data prepared successfully for per-material modeling.")
#             try:
#                 trained_models = sa.train_and_plot_models_per_material(
#                     data_df=df_full_prepared,
#                     feature_cols=model_features,
#                     target_col=model_target,
#                     material_col='Material',
#                     model_types=models_to_run,
#                     plot_feature1=plot_feature_x_3d,
#                     plot_feature2=plot_feature_y_3d,
#                     strain_rate_original_col='Strain Rate (s^-1)',
#                     output_dir=model_output_dir,
#                     scale_features=True,
#                     cv=3,
#                     min_samples_per_material=10
#                 )
#                 logging.info("Finished per-material model training and plotting attempts.")
#             except Exception as e: logging.exception(f"Error during per-material model training/plotting: {e}")
#         else: logging.error("Failed to prepare data for per-material modeling.")
#     else: logging.warning("Skipping per-material modeling: No successfully processed data available.")
# else: logging.warning("Skipping per-material modeling: Combined summary DataFrame not available.")

# --- Step 9: Combined 3D Plot (Optional, uncomment to run) ---
# logging.info("\n--- Step 9: Attempting Combined 3D Plot ---")
# if 'trained_models' in locals() and trained_models and 'df_full_prepared' in locals() and df_full_prepared is not None:
#      logging.info("Generating combined 3D surface plot for Nano and Poly materials...")
#      try:
#          materials_to_plot = ['Nano', 'Poly']
#          model_type_to_plot = 'ElasticNet'
#          models_for_plot = {}
#          for mat_label in materials_to_plot:
#              if mat_label in trained_models and model_type_to_plot in trained_models[mat_label]:
#                  models_for_plot[mat_label] = trained_models[mat_label][model_type_to_plot]
#              else: logging.warning(f"Model '{model_type_to_plot}' not found for material '{mat_label}' in trained_models dict.")
#          if not models_for_plot: logging.warning(f"No trained '{model_type_to_plot}' models found for materials: {materials_to_plot}. Skipping combined plot.")
#          else:
#              combined_3d_plot_path = os.path.join(plot_output_dir, f'COMBINED_materials_{model_type_to_plot}_3D_{processing_suffix}.png')
#              sa.plot_combined_material_surfaces(
#                  trained_models_dict=models_for_plot,
#                  data_df=df_full_prepared,
#                  feature_cols=model_features,
#                  target_col=model_target,
#                  material_col='Material',
#                  plot_feature1=plot_feature_x_3d,
#                  plot_feature2=plot_feature_y_3d,
#                  strain_rate_original_col='Strain Rate (s^-1)',
#                  output_filename=combined_3d_plot_path,
#                  color_map=sa.COLOR_MAPPING,
#                  marker_map=sa.MARKER_MAPPING
#              )
#      except AttributeError as ae:
#           if 'plot_combined_material_surfaces' in str(ae): logging.error("The 'plot_combined_material_surfaces' function is not available in spall_analysis.plotting. Skipping combined plot.")
#           else: logging.exception(f"AttributeError generating combined 3D plot: {ae}")
#      except Exception as e: logging.exception(f"Error generating combined 3D plot: {e}")
# else: logging.warning("Skipping combined 3D plot: Models were not trained or prepared data is unavailable.")


# --- Step 10: Interactive HTML Plot (Added 2024-03-19) ---
logging.info("\n--- Step 10: Generating Interactive HTML Plot ---")

try:
    # Get today's date for the filename
    today_date = datetime.now().strftime("%Y-%m-%d")
    interactive_plot_path = os.path.join(plot_output_dir, f'INTERACTIVE_spall_vs_strain_rate_{today_date}.html')
    
    # Ensure required data is available
    if combined_summary_df is None or combined_summary_df.empty:
        raise ValueError("No combined summary data available for interactive plot")
    
    # Filter data for successful processing
    plot_df = combined_summary_df[combined_summary_df['Processing Status'] == 'Success'].copy()
    if plot_df.empty:
        raise ValueError("No successfully processed data available for interactive plot")
    
    # Call the interactive plotting function
    sa.plot_interactive_spall_vs_strain_rate(
        df=plot_df,
        output_filename=interactive_plot_path,
        grain_sizes_dict=grain_sizes_to_plot,
        wilkerson_params_base=wilkerson_params_base,
        strain_rate_col='Strain Rate (s^-1)',
        spall_col='Spall Strength (GPa)',
        spall_unc_col='Spall Strength Err (GPa)',
        material_col='Material',
        filter_high_error_perc=100,
        title=f'Interactive Spall Strength vs. Strain Rate ({processing_suffix.replace("_", " ").title()})'
    )
    
    logging.info(f"Interactive plot saved to: {interactive_plot_path}")
    
except ImportError as e:
    logging.error(f"Failed to import required packages for interactive plot: {e}")
    print("Please install plotly using: pip install plotly")
except ValueError as ve:
    logging.error(f"Data validation error: {ve}")
except Exception as e:
    logging.exception(f"Error generating interactive plot: {e}")
    print(f"Failed to generate interactive plot. Check the log file for details: {log_filename}")


# --- End of Workflow ---
end_time = time.time()
logging.info(f"\n--- Spall Analysis Workflow Finished ---")
logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
logging.info(f"Results saved in: {OUTPUT_DIR}")
print(f"\n--- Workflow Finished ---")
print(f"Output saved in: {OUTPUT_DIR}")
print(f"Log saved in: {os.path.abspath(log_filename)}")

# %%
