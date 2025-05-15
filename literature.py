# spall_analysis/literature.py
"""
Functions for loading and handling literature data.
"""
import pandas as pd
import os
from . import utils # For constants if needed (e.g. shock stress calc for lit data)

def load_literature_data(filepath, required_columns=None):
    """
    Loads literature data from a CSV file.

    Args:
        filepath (str): Path to the literature CSV file.
        required_columns (list, optional): List of column names that must exist.

    Returns:
        pd.DataFrame: DataFrame containing the literature data, or None if loading fails.
    """
    if not os.path.exists(filepath):
        print(f"Error: Literature file not found: {filepath}")
        return None
    try:
        lit_df = pd.read_csv(filepath)
        print(f"Successfully loaded literature data from: {filepath}")

        # Optional: Check for required columns
        if required_columns:
            missing_cols = [col for col in required_columns if col not in lit_df.columns]
            if missing_cols:
                print(f"Warning: Literature file missing required columns: {missing_cols}")
                # Decide whether to return None or the partial data
                # return None
        
        # Optional: Perform any necessary cleaning or calculations (e.g., shock stress if needed)
        # Example: if 'Peak Velocity (m/s)' exists but 'Shock Stress (GPa)' doesn't:
        # if 'Peak Velocity (m/s)' in lit_df.columns and 'Shock Stress (GPa)' not in lit_df.columns:
        #     print("Calculating Shock Stress for literature data...")
        #     lit_df = utils.add_shock_stress_column(lit_df, velocity_col='Peak Velocity (m/s)')

        return lit_df

    except pd.errors.EmptyDataError:
        print(f"Error: Literature file is empty: {filepath}")
        return None
    except Exception as e:
        print(f"Error reading literature file {filepath}: {e}")
        return None