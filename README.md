# Spall Analysis Toolkit (`spall_analysis`)

Version: 0.1.0

A Python package designed for processing and analyzing data from spallation experiments, particularly those involving velocity interferometry (e.g., VISAR, PDV). It provides tools to calculate spall strength, strain rate, and other relevant parameters from velocity-time traces, compare results with literature data, and visualize the findings.

## Features

* **Velocity Trace Processing:** Calculates key spall parameters (peak velocity, pullback velocity, spall strength, strain rate) from raw velocity-time data using dynamic feature detection and linear fits.
* **Data Visualization:** Generates various plots:
    * Velocity trace comparisons (with optional error bands).
    * Spall Strength vs. Strain Rate (log or linear scale, with literature comparison).
    * Spall Strength vs. Shock Stress (with literature comparison).
    * (Planned/In Progress) Model comparison plots (e.g., Wilkerson model).
    * (Planned/In Progress) Elastic Net regression surface plots.
* **Modeling:** Includes implementations for:
    * (Partial) Wilkerson spall model (requires solving implicit equation).
    * Elastic Net regression for correlating spall strength with shock stress and strain rate.
* **Literature Data Integration:** Functions to load and incorporate literature data into plots.
* **Utilities:** Helper functions for file handling, unit conversions, constants, and data manipulation.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone [https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories)
    cd spall_analysis_package
    ```
    Or, download and extract the source code folder.

2.  **Install using pip:**
    It's recommended to install within a virtual environment.

    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    # Install the package and its dependencies
    pip install .
    ```
    This command reads `setup.py` and `requirements.txt` to install the package locally in editable mode (`pip install -e .` is also an option for development).

## Dependencies

The package relies on the following core libraries:

* `pandas`
* `numpy`
* `matplotlib`
* `scipy`
* `scikit-learn`
* `imageio` (Optional, for GIF generation from Elastic Net analysis)

These will be installed automatically via `pip install .`.

## Package Structure

* `spall_analysis/`: Main package source code.
    * `data_processing.py`: Core velocity trace analysis functions.
    * `plotting.py`: Plotting functions.
    * `models.py`: Material model implementations.
    * `literature.py`: Literature data handling.
    * `utils.py`: Constants, mappings, helper functions.
* `examples/`: Example scripts demonstrating usage.
* `data/`: Contains *example* literature data CSV files.
* `setup.py`: Package build script.
* `requirements.txt`: List of dependencies.
* `README.md`: This file.
* `LICENSE`: Package license (MIT).

## Usage

Import the necessary functions from the package.

```python
import spall_analysis as sa
import os

# Define input/output paths
base_dir = 'path/to/your/experiment/data' # CHANGE THIS
raw_velocity_dir = os.path.join(base_dir, 'raw_csv')
results_dir = os.path.join(base_dir, 'analysis_output')
lit_data_dir = os.path.join(base_dir, 'literature') # Or path to package's data dir

os.makedirs(results_dir, exist_ok=True)

# --- Example 1: Process Raw Velocity Traces ---

print("Processing velocity traces...")
# Assumes CSVs in 'raw_velocity_dir' have Time (ns) in col 0, Velocity (m/s) in col 1
# Adjust file_pattern as needed
summary_df = sa.process_velocity_files(
    input_folder=raw_velocity_dir,
    file_pattern='*.csv', # Adjust pattern if needed
    output_folder=os.path.join(results_dir, 'individual_trace_analysis'),
    plot_individual=True, # Generate plot for each trace
    density=sa.DENSITY_COPPER, # Use default copper density or specify another
    acoustic_velocity=sa.ACOUSTIC_VELOCITY_COPPER, # Default C0
    smooth_window=5 # Smoothing window for feature finding
)

print("\nGenerated Summary Table:")
print(summary_df.head())

# --- Example 2: Plot Spall Strength vs. Strain Rate ---

print("\nGenerating Spall vs Strain Rate plot...")
# Use the generated summary table or specific result files
# Here, we assume the summary table was saved and reload it, or use files directly
# For simplicity, let's use a pattern matching output files if generated separately
# Or better, use the summary_df directly if needed columns are present
# Assuming summary_df has 'Spall Strength (GPa)', 'Strain Rate (s^-1)', etc.

# Path to the summary table generated above
summary_table_path = os.path.join(results_dir, f"spall_summary_table_{os.path.basename(raw_velocity_dir)}.csv")
exp_files_pattern = os.path.join(results_dir,"individual_trace_analysis/*_table_*.csv") # Pattern for individual result files if needed


# Path to literature data (using the example one provided with the package)
# Find the package data directory (this is a bit complex, might need pkg_resources or importlib.resources)
# Simpler: Assume literature file is accessible at a known path
literature_file = os.path.join(lit_data_dir, 'combined_lit_table.csv') # CHANGE PATH if needed

sa.plot_spall_vs_strain_rate(
    # experimental_data_files=[summary_table_path], # Pass list with summary table path
    experimental_data_files=exp_files_pattern, # Or use pattern for individual files
    output_filename=os.path.join(results_dir, 'spall_vs_strain_rate.png'),
    literature_data_file=literature_file,
    x_col='Strain Rate (s^-1)',
    y_col='Spall Strength (GPa)',
    y_unc_col=None, # Specify column name if uncertainty data exists, e.g., 'Spall Strength Uncertainty (GPa)'
    log_scale=True,
    xlim=(1e4, 1e8), # Example limits
    ylim=(0, 7),    # Example limits
    # material_map=sa.MATERIAL_MAPPING, # Use defaults or provide custom map
    # energy_map=sa.ENERGY_VELOCITY_MAPPING, # Use defaults or provide custom map
    filter_high_error_perc=100 # Optional: filter points with >100% relative error in Y
)

# --- Example 3: Plot Spall Strength vs. Shock Stress ---

print("\nGenerating Spall vs Shock Stress plot...")
# Needs experimental data with Peak Velocity (e.g., U_fs_max(m/s)) and Spall Strength
# Can use the same summary table or individual files if they contain the velocity

literature_poly_file = os.path.join(lit_data_dir, 'combined_lit_table_only_poly.csv') # CHANGE PATH if needed

sa.plot_spall_vs_shock_stress(
    experimental_data_files=exp_files_pattern, # Pattern for individual files
    output_filename=os.path.join(results_dir, 'spall_vs_shock_stress.png'),
    literature_data_file=literature_poly_file,
    exp_vel_col='U_fs_max(m/s)', # Column containing peak velocity in exp files
    spall_col='Spall Strength (GPa)',
    spall_unc_col=None, # Specify uncertainty column if available
    density=sa.DENSITY_COPPER,
    acoustic_velocity=sa.ACOUSTIC_VELOCITY_COPPER,
    xlim=(0, 15), # Example limits
    ylim=(0, 7),   # Example limits
    filter_high_error_perc=100 # Optional filtering
)

# --- Example 4: Train Elastic Net Model ---
# (Requires data loaded appropriately, e.g., into a dictionary)

print("\nTraining Elastic Net model...")
# Assuming you have loaded individual result CSVs into a dictionary like:
# data_dict_4um = {'800mJ': pd.read_csv(...), '1200mJ': pd.read_csv(...)}
# data_dict_100nm = {...}
# combined_dict = {**data_dict_4um, **data_dict_100nm} # Or process separately

# This part requires loading the specific CSVs mentioned in script 10
# Placeholder for loading logic:
data_dict = {}
required_files = [ # Example files needed by script 10
    '4um_spall_strength_strain_rate_table_800mJ.csv',
    '4um_spall_strength_strain_rate_table_1200mJ.csv',
    '4um_spall_strength_strain_rate_table_1350mJ.csv',
    '4um_spall_strength_strain_rate_table_1500mJ.csv',
    '100nm_spall_strength_strain_rate_table_800mJ.csv',
    '100nm_spall_strength_strain_rate_table_1000mJ.csv',
    '100nm_spall_strength_strain_rate_table_1200mJ.csv',
    '100nm_spall_strength_strain_rate_table_1500mJ.csv',
    '100nm_spall_strength_strain_rate_table_1700mJ.csv'
]
# Assume these files are in 'results_dir' or another known location
files_found = True
for fname in required_files:
    fpath = os.path.join(results_dir, fname) # Adjust path if needed
    if os.path.exists(fpath):
         data_dict[fname.replace('.csv','')] = pd.read_csv(fpath)
    else:
         print(f"Warning: Required file for Elastic Net not found: {fpath}")
         files_found = False

if files_found and data_dict:
    # Prepare features (X) and target (y)
    # Ensure correct column names are used - check your CSVs!
    # Script 10 used 'Peak Shock Stress (GPa)' and 'Scaled Strain Rate...'
    # Let's try calculating shock stress and using the standard names.
    X_combined, y_combined = sa.prepare_feature_matrix(
        data_dict=data_dict,
        # Adjust column names based on your actual CSV output from process_velocity_files
        shock_stress_col='Shock Stress (GPa)', # Calculated if missing
        strain_rate_col='Strain Rate (s^-1)',
        spall_col='Spall Strength (GPa)',
        velocity_col='U_fs_max(m/s)', # Used for shock stress calculation
        scale_strain_rate=1e-6 # Match scaling used in original script 10
    )

    if X_combined is not None:
        # Train the model
        model, scaler, X_scaled, train_r2, _ = sa.train_elastic_net(
            X=X_combined,
            y=y_combined,
            scale_features=True,
            cv=5
        )

        # Add plotting of results here if desired, using functions from plotting.py
        # sa.plot_elastic_net_results(model, scaler, X_scaled, y_combined, ...) # Placeholder

else:
    print("Skipping Elastic Net example due to missing input files.")


print("\nAnalysis complete. Check the '{results_dir}' directory for outputs.")