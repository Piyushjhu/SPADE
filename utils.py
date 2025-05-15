# spall_analysis/utils.py
"""
Utility functions and constants for the spall_analysis package.

(Corrected extract_legend_info to return material type even if energy is not found)
"""
import os
import pandas as pd
import numpy as np
import glob
import logging # Use logging for warnings

# --- Constants ---
DENSITY_COPPER = 8960.0 # kg/m^3 (Example, adjust if needed)
ACOUSTIC_VELOCITY_COPPER = 3940.0 # m/s (Example, adjust if needed)

# --- Mappings ---
# These should be defined by the user based on their file/folder naming conventions
MATERIAL_MAPPING = {
    "4um": "Poly",                 # Polycrystalline
    "100nm": "Nano",               # Nanocrystalline
    "SC100": "SC [100]",           # Single Crystal [100]
    "SC110": "SC [110]",           # Single Crystal [110]
    "SC111": "SC [111]",           # Single Crystal [111]
    # Add others as needed, ensure keys match folder prefixes
}

ENERGY_VELOCITY_MAPPING = {
    "a mJ": "aa m/s",
    "b mJ": "bb m/s",
    "c mJ": "cc m/s",
    "d mJ": "dd m/s",
    "e mJ": "ee m/s",
    "f mJ": "ff m/s",
    "g mJ": "gg m/s",
    "h mJ": "hh m/s"
    # Add others as needed
}

COLOR_MAPPING = {
    "Nano": "r",  # Red for Nano
    "Poly": "c",  # Cyan for Poly
    "SC [100]": "m",    # Magenta
    "SC [110]": "g",    # Green
    "SC [111]": "b",    # Blue
    "Default": "gray", # Fallback color for experimental data if type unknown
    # Add colors for literature sources or specific model lines
    "aaa": "#1f77b4", # Example color
    "bbb": "#ff7f0e", # Example color
    "ccc": "#2ca02c", # Example color
    "ddd": "#9467bd", # Example color
    "eee": "#8c564b", # Example color
    "Arad et al.": "#e377c2", # Example color
    "Escobedo et al. ": "#7f7f7f", # Example color
    "Fortov et al.": "#bcbd22", # Example color
    "G. Kanel et al. ": "#17becf", # Example color
    "Minich et al. ": "#aec7e8", # Example color
    "Mukherjee et al. ": "#ffbb78", # Example color
    "Ogoronikov et al. ": "#98df8a", # Example color
    "Paisley et al. ": "#ff9896", # Example color
    "Peralta et al.": "#c5b0d5", # Example color
    "T.Chen et al. (": "#c49c94", # Example color
    "Turney et al. ": "#f7b6d2", # Example color
    "Yong-Gang et al. (nano)": "#dbdb8d", # Example color
    # Model lines from multi-wilkerson plot
    "Poly (4um)": "#2ca02c", # Green solid
    "Nano (100nm)": "#d62728", # Red dashed
    "Single Crystal (Est. dG=1mm)": "#9467bd", # Purple dotted
}

MARKER_MAPPING = {
    "Nano": "s", # Square for Nano (Changed from ^)
    "Poly": "o", # Circle for Poly
    "SC [100]": "D",    # Diamond
    "SC [110]": "P",    # Plus (filled)
    "SC [111]": "X",    # X (filled)
    "Default": "x",   # Fallback marker for experimental data
     # Define markers for literature sources
    "Kanel": "v",
    "Moshe": "D",
    "Chen": "p",
    "Wilkerson": "h",
    "Priyadarshan": "*",
    "Arad et al. (poly)": "d",
    "Escobedo et al. (poly)": "<",
    "Fortov et al.": ">",
    "G. Kanel et al. (poly)": "1",
    "Minich et al. (single Crystal)": "2",
    "Mukherjee et al. (poly)": "3",
    "Ogoronikov et al. (poly)": "4",
    "Paisley et al. (poly)": "8",
    "Peralta et al. (poly)": "P",
    "T.Chen et al. (poly)": "X",
    "Turney et al. (Single Crystal)": "H",
    "Yong-Gang et al. (nano)": "+",
}


# --- Utility Functions ---

def find_data_files(input_dir, file_pattern):
    """ Finds files matching a pattern in a directory using glob. """
    search_path = os.path.join(input_dir, file_pattern)
    files = sorted(glob.glob(search_path))
    if not files:
        logging.warning(f"No files found matching '{search_path}'")
    else:
        logging.info(f"Found {len(files)} files matching '{search_path}'")
    return files

def extract_legend_info(filename, material_map=MATERIAL_MAPPING, energy_map=ENERGY_VELOCITY_MAPPING):
    """
    Extracts material type and velocity label from a filename or path basename.

    Args:
        filename (str): The input filename or path.
        material_map (dict): Mapping from filename prefix to material name.
        energy_map (dict): Mapping from energy string (e.g., '800mJ') to velocity string.

    Returns:
        tuple: (material_type, velocity_label, energy_key)
               Values can be None if not found.
    """
    if material_map is None: material_map = {}
    if energy_map is None: energy_map = {}

    # Use basename to handle full paths passed from process_velocity_files
    base = os.path.basename(filename).replace('.csv', '')
    parts = base.split('_') # Split by underscore, common convention

    material_type = None
    velocity_label = None
    energy_key = None

    # Find material based on prefix matching keys in material_map
    # Check against the whole base name first
    for prefix, mat_type in material_map.items():
        if base.startswith(prefix):
            material_type = mat_type
            logging.debug(f"DEBUG extract_legend: Matched prefix '{prefix}' in '{base}' -> '{mat_type}'")
            break # Take first match

    # If no prefix match, maybe check parts? (Optional, depends on naming)
    # if material_type is None and len(parts) > 0:
    #    if parts[0] in material_map:
    #         material_type = material_map[parts[0]]
    #         logging.debug(f"DEBUG extract_legend: Matched part '{parts[0]}' in '{base}' -> '{material_type}'")

    # Find energy/velocity based on keys in energy_map
    # Check if the energy key exists as a distinct part or within any part
    found_energy = False
    for energy, vel_label in energy_map.items():
        if energy in parts: # Check if energy key is a whole part
             velocity_label = vel_label
             energy_key = energy
             logging.debug(f"DEBUG extract_legend: Matched energy part '{energy}' in '{base}' -> '{vel_label}'")
             found_energy = True
             break
        # Also check if energy key is contained within any part (e.g., "800mJ_run1")
        for part in parts:
             if energy in part:
                 velocity_label = vel_label
                 energy_key = energy
                 logging.debug(f"DEBUG extract_legend: Matched energy substring '{energy}' in part '{part}' of '{base}' -> '{vel_label}'")
                 found_energy = True
                 break
        if found_energy: break


    # ** MODIFIED RETURN LOGIC **
    # Log warnings if parts couldn't be parsed
    if material_type is None:
        logging.warning(f"Could not parse material type reliably from '{base}' using MATERIAL_MAPPING.")
    if velocity_label is None:
         logging.warning(f"Could not parse energy/velocity label reliably from '{base}' using ENERGY_VELOCITY_MAPPING.")

    # Return the tuple, allowing for None values
    return material_type, velocity_label, energy_key


def calculate_shock_stress(velocity, density, acoustic_velocity):
    """ Calculates Hugoniot shock stress (in GPa). """
    if pd.isna(velocity) or density <= 0 or acoustic_velocity <= 0:
        return np.nan
    # Formula: P = 0.5 * rho_0 * C_0 * u_p (where u_p is particle velocity = 0.5 * u_fs)
    # Shock stress (GPa) = 0.5 * density (kg/m^3) * acoustic_velocity (m/s) * velocity (m/s) * 1e-9
    return 0.5 * density * acoustic_velocity * velocity * 1e-9

def add_shock_stress_column(df, velocity_col, density, acoustic_velocity, new_col_name='Shock Stress (GPa)'):
    """ Adds a shock stress column to a DataFrame based on a velocity column. """
    if velocity_col not in df.columns:
        logging.error(f"Velocity column '{velocity_col}' not found in DataFrame.")
        return df
    df[new_col_name] = df[velocity_col].apply(
        lambda v: calculate_shock_stress(v, density, acoustic_velocity)
    )
    logging.info(f"Added '{new_col_name}' column based on '{velocity_col}'.")
    return df

# --- Add other utility functions from previous versions if needed ---

