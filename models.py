# spade_analysis/models.py
"""
Implementation of material models relevant to spall analysis using SPADE.
Includes complex Wilkerson model based on script 6_b and Elastic Net.

(Updated train_and_plot_models_per_material for specific scaling and features)
"""

import pandas as pd
import numpy as np
import os
from scipy.optimize import fsolve
from scipy.special import gamma as gamma_func # Rename to avoid conflict
from sklearn.linear_model import ElasticNetCV, LassoCV # Added LassoCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler # Import both scalers
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
import matplotlib.pyplot as plt
import warnings
import math # Needed for Wilkerson helpers
import logging # Use logging
from . import utils # Import utilities for constants or helper functions

# --- Complex Wilkerson Model Functions (from script 6_b) ---
# ... (Wilkerson functions remain the same) ...
def _zeta_alpha(beta_alpha):
    """ Helper for Wilkerson model: Calculates zeta_alpha based on beta_alpha. """
    if beta_alpha == 1: return 1.0
    else:
        try: upper_limit = math.floor(beta_alpha + 0.5)
        except TypeError: return np.inf # Handle non-numeric beta_alpha
        if upper_limit < 1: return 1.0
        product_term = 1.0
        for i in range(1, upper_limit + 1):
             term = (9.0 + 2.0 * i)
             if term == 0: return np.inf
             product_term *= (1.0 / term)
        try: return (1.0 / gamma_func(beta_alpha + 4.5)) * product_term
        except (ValueError, OverflowError): return np.inf

def _calculate_sigma_th(strain_rate, beta_alpha, zeta_alpha, rho, E_pa, K0_pa):
    """ Helper for Wilkerson model: Calculates sigma_th. """
    if zeta_alpha == 0 or np.isinf(zeta_alpha): return np.zeros_like(strain_rate) # Avoid division by zero or inf
    term1 = (strain_rate / (1.0e6 * zeta_alpha))
    # Handle potential negative values before power for safety
    term1[term1 < 0] = 0
    try:
        sigma_th_pa = (rho * E_pa * K0_pa)**(1/3) * term1**(1/3)
        return sigma_th_pa
    except Exception as e:
        logging.warning(f"Could not calculate sigma_th: {e}")
        return np.full_like(strain_rate, np.nan)

def _calculate_sigma_y(dG, sigma0_pa, ky_sqrtm):
    """ Helper for Wilkerson model: Calculates sigma_y using Hall-Petch. """
    if dG <= 0: return np.inf # Avoid division by zero or invalid sqrt
    return sigma0_pa + ky_sqrtm / np.sqrt(dG)

def _calculate_spall_strength_complex(strain_rate, dG, sigma0_pa, ky_sqrtm, E_pa, Reos_pa, K0_pa, rho, N2, N0_GB, d0_G):
    """ Calculates spall strength using the complex Wilkerson model. """
    beta_alpha = N2 * dG / (N0_GB * d0_G)
    zeta_alpha = _zeta_alpha(beta_alpha)
    sigma_y_pa = _calculate_sigma_y(dG, sigma0_pa, ky_sqrtm)
    sigma_th_pa = _calculate_sigma_th(strain_rate, beta_alpha, zeta_alpha, rho, E_pa, K0_pa)
    # Ensure calculations handle potential NaNs or Infs safely
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore potential overflow/invalid value in power
        try:
            spall_strength_pa = sigma_y_pa * (1.0 + (sigma_th_pa / sigma_y_pa)**(2/3))**(3/2)
        except Exception as e:
            logging.warning(f"Error in spall strength calculation step: {e}")
            return np.full_like(strain_rate, np.nan)
    return spall_strength_pa

def calculate_wilkerson_spall_complex(strain_rate, dG, sigma0_pa, ky_sqrtm, E_pa, Reos_pa, K0_pa, rho, N2, N0_GB, d0_G):
    """ Vectorized calculation of complex Wilkerson model spall strength. """
    strain_rate_arr = np.asarray(strain_rate)
    if np.any(strain_rate_arr <= 0):
        logging.warning("Strain rates must be positive for Wilkerson model.")
        # Handle non-positive strain rates, e.g., return NaN or raise error
        results = np.full_like(strain_rate_arr, np.nan, dtype=float)
        positive_mask = strain_rate_arr > 0
        if np.any(positive_mask):
             results[positive_mask] = _calculate_spall_strength_complex(strain_rate_arr[positive_mask], dG, sigma0_pa, ky_sqrtm, E_pa, Reos_pa, K0_pa, rho, N2, N0_GB, d0_G)
        return results
    else:
        return _calculate_spall_strength_complex(strain_rate_arr, dG, sigma0_pa, ky_sqrtm, E_pa, Reos_pa, K0_pa, rho, N2, N0_GB, d0_G)


# --- Feature Preparation (Keep the modified version from previous step) ---
def prepare_feature_matrix(data_dict,
                           feature_cols,
                           target_col,
                           required_cols=None,
                           metadata_cols_to_keep=None, # *** Keep NEW ARGUMENT ***
                           calculate_shock_stress_if_missing=True,
                           velocity_col='First Maxima (m/s)', # Use the alias
                           density=utils.DENSITY_COPPER,
                           acoustic_velocity=utils.ACOUSTIC_VELOCITY_COPPER,
                           transformations=None
                          ):
    """
    Combines data and prepares feature matrix (X) and target vector (y).
    Also keeps specified metadata columns in the returned DataFrame.
    """
    # ... (Keep the implementation that retains metadata_cols_to_keep) ...
    all_processed_dfs = []
    final_feature_names = list(feature_cols) # Ensure it's a list
    if metadata_cols_to_keep is None:
        metadata_cols_to_keep = []
    else:
        metadata_cols_to_keep = list(metadata_cols_to_keep)

    logging.info("Preparing feature matrix...")
    for label, df_orig in data_dict.items():
        logging.info(f" Processing data source: {label}")
        df = df_orig.copy()
        df_processed = pd.DataFrame(index=df.index)
        df_processed['source_label'] = label

        base_cols_needed = set(required_cols or [])
        if transformations:
            for input_col in transformations:
                base_cols_needed.add(input_col)
        # *** Use 'Peak Shock Pressure (GPa)' as the target name for calculation check ***
        shock_stress_col_actual = 'Peak Shock Pressure (GPa)'
        if calculate_shock_stress_if_missing and shock_stress_col_actual in final_feature_names:
             base_cols_needed.add(velocity_col)
        base_cols_needed.update(final_feature_names)
        base_cols_needed.add(target_col)
        base_cols_needed.update(metadata_cols_to_keep)

        actual_required = list(base_cols_needed.intersection(df.columns))
        missing_base = [col for col in (required_cols or []) if col not in df.columns]
        if missing_base:
            logging.warning(f"  Skipping '{label}': Missing required base columns: {missing_base}")
            continue

        cols_to_copy_initially = list(base_cols_needed.intersection(df.columns))
        for col in cols_to_copy_initially:
             is_output_of_transform = transformations and any(t_info.get('out_name') == col for t_info in transformations.values())
             # *** Check against shock_stress_col_actual ***
             is_shock_stress_to_calc = calculate_shock_stress_if_missing and col == shock_stress_col_actual and col not in df.columns

             if not is_output_of_transform and not is_shock_stress_to_calc:
                 df_processed[col] = df[col]
                 logging.debug(f"  Copied base column: {col}")

        # --- Calculate Shock Stress if needed ---
        shock_stress_requested = shock_stress_col_actual in final_feature_names or shock_stress_col_actual in metadata_cols_to_keep
        if shock_stress_requested and shock_stress_col_actual not in df_processed.columns: # Check df_processed now
            if calculate_shock_stress_if_missing and velocity_col in df.columns: # Check original df for velocity
                logging.info(f"  Calculating '{shock_stress_col_actual}' using '{velocity_col}'.")
                temp_df_for_calc = df[[velocity_col]].copy()
                # *** Use the correct new_col_name ***
                temp_df_for_calc = utils.add_shock_stress_column(temp_df_for_calc, velocity_col=velocity_col,
                                                                  density=density, acoustic_velocity=acoustic_velocity,
                                                                  new_col_name=shock_stress_col_actual)
                if shock_stress_col_actual in temp_df_for_calc.columns:
                    df_processed[shock_stress_col_actual] = temp_df_for_calc[shock_stress_col_actual]
                else:
                     logging.warning(f"  Skipping '{label}': Failed to calculate '{shock_stress_col_actual}'.")
                     continue
            elif calculate_shock_stress_if_missing and velocity_col not in df.columns:
                 logging.warning(f"  Skipping '{label}': Cannot calculate '{shock_stress_col_actual}' because base column '{velocity_col}' is missing.")
                 continue
            else: # Shock stress requested, missing, and calculation disabled/impossible
                logging.warning(f"  Skipping '{label}': Column '{shock_stress_col_actual}' requested but missing, and calculation disabled or base column missing.")
                continue

        # --- Apply Transformations ---
        if transformations:
            logging.info(f"  Applying transformations...")
            for input_col, transform_info in transformations.items():
                output_col = transform_info.get('out_name')
                transform_func = transform_info.get('func')
                if not output_col or not callable(transform_func):
                     logging.warning(f"  Invalid transformation spec for input '{input_col}' in '{label}'. Skipping.")
                     continue
                if input_col in df_processed.columns or input_col in df.columns:
                    input_data = df_processed[input_col] if input_col in df_processed.columns else df[input_col]
                    try:
                        df_processed[output_col] = transform_func(input_data)
                        logging.info(f"    Applied transformation: '{input_col}' -> '{output_col}'")
                        # Add the new transformed column to the list of final feature names if it's requested
                        if output_col in feature_cols and output_col not in final_feature_names:
                             final_feature_names.append(output_col)
                             logging.debug(f"    Added transformed column '{output_col}' to final features.")
                    except Exception as e:
                         logging.warning(f"  Failed to apply transformation '{input_col}' -> '{output_col}' for '{label}': {e}")
                else:
                     logging.warning(f"  Input column '{input_col}' for transformation not found in '{label}'. Cannot create '{output_col}'.")


        # --- Check if all FINAL requested columns (features + target + metadata) are present ---
        final_cols_required = final_feature_names + [target_col] + metadata_cols_to_keep
        missing_final_cols = [col for col in final_cols_required if col not in df_processed.columns]

        if missing_final_cols:
            logging.warning(f"  Skipping '{label}': Missing final required columns after processing: {missing_final_cols}. Present columns: {df_processed.columns.tolist()}")
            continue

        all_processed_dfs.append(df_processed[final_cols_required + ['source_label']])

    # --- Combine and Final Cleanup ---
    if not all_processed_dfs:
        logging.error("\nError: No valid data found to combine after processing all sources.")
        return None, None, None

    combined_df = pd.concat(all_processed_dfs, ignore_index=True)
    logging.info(f"\nCombined data from {len(all_processed_dfs)} sources. Initial rows: {len(combined_df)}")

    numeric_cols = final_feature_names + [target_col]
     # <<< INSERT DEBUGGING CODE HERE >>>
    logging.debug(f"Columns in combined_df before numeric conversion: {combined_df.columns.tolist()}")
    logging.debug(f"Duplicate columns check: {combined_df.columns[combined_df.columns.duplicated()].tolist()}") # Check for duplicates
    logging.debug(f"Columns to convert to numeric: {numeric_cols}")
    logging.debug(f"Data types before conversion:\n{combined_df[numeric_cols].info()}")

    for col in numeric_cols:
        if col in combined_df.columns:
             logging.debug(f"Attempting pd.to_numeric on column: '{col}'")
             logging.debug(f"Type of combined_df['{col}']: {type(combined_df[col])}")
             # Add this check:
             if not isinstance(combined_df[col], (pd.Series, list, tuple, np.ndarray)):
                 logging.error(f"Column '{col}' is not a Series, list, tuple, or ndarray. Type is {type(combined_df[col])}. Skipping conversion.")
                 try:
                     # Log head if it's a DataFrame (likely cause)
                     logging.error(f"Content sample (potential DataFrame):\n{combined_df[col].head().to_string()}")
                 except AttributeError:
                     logging.error(f"Content sample (unknown type): {str(combined_df[col])[:200]}...") # Log beginning if unknown type
                 continue # Skip to next column

             # Original conversion attempt
             combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce') # Line 216
             logging.debug(f"  Converted column '{col}' to numeric.")
        else:
             logging.error(f"  Column '{col}' expected but not found in combined data for numeric conversion.")
             return None, None, None
        
    # for col in numeric_cols:
    #     if col in combined_df.columns:
    #          combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    #          logging.debug(f"  Converted column '{col}' to numeric.")
    #     else:
    #          logging.error(f"  Column '{col}' expected but not found in combined data for numeric conversion.")
    #          return None, None, None

    initial_rows = len(combined_df)
    combined_df.dropna(subset=numeric_cols, inplace=True)
    removed_rows = initial_rows - len(combined_df)
    if removed_rows > 0:
        logging.info(f"Removed {removed_rows} rows with NaN values in final features or target.")

    if combined_df.empty:
        logging.error("Error: Combined DataFrame is empty after dropping NaNs from features/target.")
        return None, None, None

    X_combined = combined_df[final_feature_names]
    y_combined = combined_df[target_col]
    logging.info(f"Prepared feature matrix X with columns {X_combined.columns.tolist()} and shape: {X_combined.shape}")
    logging.info(f"Prepared target vector y ('{target_col}') with shape: {y_combined.shape}")
    logging.info(f"Returned combined DataFrame with columns {combined_df.columns.tolist()} and shape: {combined_df.shape}")

    return X_combined, y_combined, combined_df


# --- Original Elastic Net Trainer (can be kept or removed if unused) ---
def train_elastic_net(X, y, scale_features=True, cv=5, l1_ratios=[.1, .5, .7, .9, .95, .99, 1], random_state=42, max_iter=10000):
    """ Trains a single Elastic Net regression model using cross-validation (ElasticNetCV). """
    # ... (implementation remains the same) ...
    X_proc = X.copy(); scaler = None
    logging.info("\nTraining Elastic Net model...")
    if scale_features:
        logging.info(" Scaling features using MinMaxScaler.")
        scaler = MinMaxScaler()
        try:
            X_proc_scaled = scaler.fit_transform(X_proc)
            if isinstance(X_proc, pd.DataFrame): X_proc = pd.DataFrame(X_proc_scaled, columns=X_proc.columns, index=X_proc.index)
            else: X_proc = X_proc_scaled
        except Exception as e: logging.warning(f"Feature scaling failed: {e}. Proceeding without scaling."); scaler = None; X_proc = X.copy()
    else: logging.info(" Skipping feature scaling.")
    try:
        model = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, random_state=random_state, n_alphas=100, max_iter=max_iter, n_jobs=-1, tol=1e-4)
        model.fit(X_proc, y)
        logging.info("\n--- Elastic Net Training Summary ---")
        logging.info(f"Best Alpha (Regularization Strength): {model.alpha_:.5f}")
        logging.info(f"Best L1 Ratio (Mixing Parameter): {model.l1_ratio_:.2f}")
        if hasattr(model, 'n_iter_'): logging.info(f"Number of Iterations: {model.n_iter_}")
        logging.info("\nCoefficients:")
        if isinstance(X_proc, pd.DataFrame): logging.info("\n" + pd.Series(model.coef_, index=X_proc.columns).to_string())
        else: logging.info(model.coef_)
        logging.info(f"\nIntercept: {model.intercept_:.4f}")
        y_pred_train = model.predict(X_proc); r2_train = r2_score(y, y_pred_train); rmse_train = np.sqrt(mean_squared_error(y, y_pred_train))
        logging.info(f"\nPerformance on Training Data:"); logging.info(f"  R-squared: {r2_train:.4f}"); logging.info(f"  RMSE: {rmse_train:.4f}"); logging.info("------------------------------------")
        return model, scaler, X_proc, r2_train, rmse_train
    except Exception as e:
        logging.exception(f"Error during Elastic Net training: {e}")
        return None, scaler, X_proc, None, None


# --- ** UPDATED FUNCTION: Train and Plot Models Per Material with Specific Scaling ** ---
def train_and_plot_models_per_material(
    data_df, # Combined DataFrame with features, target, and Material column
    feature_cols, # List of feature column names to use FOR MODELING (e.g., ['Peak Shock Pressure (GPa)', 'Scaled Strain Rate'])
    target_col,   # Name of the target column
    material_col, # Name of the column identifying material type
    model_types,  # List of strings: e.g., ['ElasticNet', 'Lasso']
    plot_feature1, # Feature name for 3D plot X-axis (e.g., 'Peak Shock Pressure (GPa)')
    plot_feature2, # Feature name for 3D plot Y-axis (e.g., 'Scaled Strain Rate')
    strain_rate_original_col, # Name of the ORIGINAL strain rate column used for scaling
    output_dir,    # Base directory to save plots (e.g., 'models' subdir)
    scale_features=True, # Kept for consistency, but logic now specific
    cv=3, # Reduced CV folds for smaller datasets per material
    random_state=42,
    min_samples_per_material=10 # Minimum samples needed to train for a material
    ):
    """
    Trains specified regression models for each material type subset using specific
    scaling (MinMaxScaler on strain rate only) and saves 3D plots.

    Args:
        data_df (pd.DataFrame): DataFrame containing features, target, and material_col.
                                MUST contain strain_rate_original_col and plot_feature1.
        feature_cols (list): List of column names to use as features for the model.
                             Should typically include plot_feature1 and the SCALED strain rate name.
        target_col (str): Name of the target variable column.
        material_col (str): Column name identifying material types.
        model_types (list): List of model names ('ElasticNet', 'Lasso').
        plot_feature1 (str): Feature name for the X-axis of the 3D plot (e.g., 'Peak Shock Pressure (GPa)').
        plot_feature2 (str): Feature name for the Y-axis of the 3D plot (e.g., 'Scaled Strain Rate').
        strain_rate_original_col (str): The name of the original, unscaled strain rate column.
        output_dir (str): Directory path to save the generated plots.
        scale_features (bool): If True, applies MinMaxScaler to strain_rate_original_col.
        cv (int): Number of cross-validation folds for CV models.
        random_state (int): Random seed for reproducibility.
        min_samples_per_material (int): Minimum data points required for a material to train.

    Returns:
        dict: Dictionary containing trained models and scalers, keyed by (material, model_name).
              Example: {'Nano': {'ElasticNet': {'model': obj, 'scaler': obj}, ...}, ...}
    """
    logging.info("\n--- Training and Plotting Models Per Material (Specific Scaling) ---")
    # *** Import moved inside function to avoid circular dependency ***
    from .plotting import plot_model_3d_surface

    if material_col not in data_df.columns:
        logging.error(f"Material column '{material_col}' not found in DataFrame. Aborting.")
        return {}
    if strain_rate_original_col not in data_df.columns:
         logging.error(f"Original strain rate column '{strain_rate_original_col}' not found in DataFrame. Aborting.")
         return {}
    if plot_feature1 not in data_df.columns:
         logging.error(f"Plot feature 1 column '{plot_feature1}' not found in DataFrame. Aborting.")
         return {}
    # Check if the target scaled strain rate column exists (it should have been created by prepare_feature_matrix)
    scaled_strain_rate_col = None
    for col in feature_cols:
        if 'Scaled Strain Rate' in col:
            scaled_strain_rate_col = col
            break
    if scaled_strain_rate_col is None or scaled_strain_rate_col not in data_df.columns:
         logging.error(f"Scaled strain rate column (expected in {feature_cols}) not found in DataFrame. Aborting.")
         return {}


    # Ensure output directory exists
    model_plot_dir = os.path.join(output_dir, 'model_plots_per_material')
    os.makedirs(model_plot_dir, exist_ok=True)
    logging.info(f"Plots will be saved in: {model_plot_dir}")

    # Use a nested dictionary structure: {material: {model_type: {details}}}
    trained_models_dict = {}

    for material_label in data_df[material_col].unique():
        logging.info(f"\nProcessing Material: {material_label}")
        material_df = data_df[data_df[material_col] == material_label].copy()
        material_df = material_df.dropna(subset=feature_cols + [target_col]) # Drop NaNs relevant to modeling

        if len(material_df) < min_samples_per_material:
            logging.warning(f"Skipping material '{material_label}': Insufficient samples after NaN drop ({len(material_df)} < {min_samples_per_material}).")
            continue

        trained_models_dict[material_label] = {} # Initialize dict for this material

        # --- Prepare Features for Modeling (Peak Shock Pressure + Scaled Strain Rate) ---
        # Select the columns specified in feature_cols
        X_mat = material_df[feature_cols].copy()
        y_mat = material_df[target_col]

        # The 'Scaled Strain Rate' column should already exist from prepare_feature_matrix.
        # The 'Peak Shock Pressure (GPa)' column is used directly (unscaled).
        # No additional scaling needed here if prepare_feature_matrix handled it.
        # We still need a scaler object representing the scaling applied to strain rate *for plotting*.
        strain_rate_scaler = None
        if scale_features: # Fit a scaler *only* on the original strain rate for this material subset
            strain_rate_scaler = MinMaxScaler()
            try:
                # Fit scaler on the original strain rate column for this material
                strain_rate_scaler.fit(material_df[[strain_rate_original_col]])
                logging.info(f" Fitted MinMaxScaler on '{strain_rate_original_col}' for {material_label}.")
                # Verify the 'Scaled Strain Rate' column matches this scaling (optional check)
                # scaled_check = strain_rate_scaler.transform(material_df[[strain_rate_original_col]])
                # if not np.allclose(material_df[scaled_strain_rate_col].values, scaled_check.flatten()):
                #      logging.warning(f" Pre-calculated '{scaled_strain_rate_col}' might not match fresh scaling for {material_label}.")
            except Exception as e:
                 logging.warning(f" Failed to fit MinMaxScaler for {material_label}: {e}. Scaler will be None.")
                 strain_rate_scaler = None

        # --- Model Training Loop ---
        for model_name in model_types:
            logging.info(f" Training {model_name} model for {material_label}...")
            model = None
            try:
                if model_name == 'ElasticNet':
                    model = ElasticNetCV(cv=cv, random_state=random_state, n_alphas=100, max_iter=10000, n_jobs=-1, tol=1e-4)
                elif model_name == 'Lasso':
                    model = LassoCV(cv=cv, random_state=random_state, n_alphas=100, max_iter=10000, n_jobs=-1, tol=1e-4)
                else:
                    logging.warning(f" Unknown model type '{model_name}' requested. Skipping.")
                    continue

                # Train on the selected features (e.g., Peak Shock Pressure, Scaled Strain Rate)
                model.fit(X_mat, y_mat)

                # Log summary
                logging.info(f"  --- {model_name} ({material_label}) Training Summary ---")
                if hasattr(model, 'alpha_'): logging.info(f"  Best Alpha: {model.alpha_:.5f}")
                if hasattr(model, 'l1_ratio_'): logging.info(f"  Best L1 Ratio: {model.l1_ratio_:.2f}")
                if hasattr(model, 'n_iter_'): logging.info(f"  Number of Iterations: {model.n_iter_}")
                logging.info("  Coefficients:")
                logging.info("\n" + pd.Series(model.coef_, index=X_mat.columns).to_string()) # Use X_mat columns
                logging.info(f"  Intercept: {model.intercept_:.4f}")
                y_pred_mat_train = model.predict(X_mat)
                r2_train = r2_score(y_mat, y_pred_mat_train)
                rmse_train = np.sqrt(mean_squared_error(y_mat, y_pred_mat_train))
                logging.info(f"  Performance on Training Data ({material_label}): R²={r2_train:.4f}, RMSE={rmse_train:.4f}")
                logging.info(f"  -----------------------------------------")

                # Store model and the *strain rate only* scaler
                trained_models_dict[material_label][model_name] = {'model': model, 'scaler': strain_rate_scaler}

                # --- Generate 3D Plot for this model ---
                logging.info(f"  Generating 3D plot for {model_name} ({material_label})...")
                plot_filename_base = f"model_plot_{material_label.replace(' ', '_').replace('[','').replace(']','')}_{model_name}_3D.png"
                plot_save_path = os.path.join(model_plot_dir, plot_filename_base)

                # Check if plot features exist in the original material_df (before scaling)
                if plot_feature1 not in material_df.columns or plot_feature2 not in material_df.columns:
                     logging.warning(f"  Skipping 3D plot: Required plot features ('{plot_feature1}', '{plot_feature2}') not found in original data for {material_label}.")
                     continue

                try:
                    plot_model_3d_surface(
                        model=model,
                        scaler=strain_rate_scaler, # Pass the scaler fitted ONLY on strain rate
                        X_original_df=material_df, # Pass the original unscaled data for this material
                        y_actual=y_mat,
                        feature1_name=plot_feature1, # e.g., 'Peak Shock Pressure (GPa)'
                        feature2_name=plot_feature2, # e.g., 'Scaled Strain Rate'
                        all_feature_names=feature_cols, # Features the model was trained on
                        output_filename=plot_save_path,
                        material_col=material_col,
                        title=f'{model_name} Prediction ({material_label})\n{plot_feature1} vs {plot_feature2}',
                        color_map=utils.COLOR_MAPPING,
                        marker_map=utils.MARKER_MAPPING,
                        # Pass the original strain rate column name IF needed by plotting func for scaling grid
                        # original_strain_rate_col_for_plot=strain_rate_original_col
                    )
                except Exception as plot_e:
                    logging.exception(f"  Error generating 3D plot for {model_name} ({material_label}): {plot_e}")

            except Exception as train_e:
                logging.exception(f"Error training {model_name} model for {material_label}: {train_e}")

    logging.info("\n--- Finished Training and Plotting Models Per Material ---")
    return trained_models_dict

#=============================================================================#
# == FUNCTIONS BASED ON Wilkerson & Ramesh, PRL 117, 215503 (2016) == #
#=============================================================================#


# import math
# from scipy.special import gamma as gamma_func
# import numpy as np
# import pandas as pd
# import logging

def _Ry_paper(sigma0_pa, ky_sqrtm, E_pa, dG):
    """ Calculates Ry (cavitation initiation pressure) using Eq. (2) from PRL paper. """
    # Using np.isclose to handle potential floating point comparisons with 0
    if np.isclose(dG, 0) or dG < 0: # Treat dG=0 or negative as invalid for polycrystal calculation
         logging.debug("Calculating Ry for effective single crystal (using sigma0)")
         # Use the limit for large dG where sigma_y -> sigma0
         sigma_y_eff = sigma0_pa
    elif np.isinf(dG): # Handle explicit infinity for single crystal case
         logging.debug("Calculating Ry for explicit single crystal (dG=inf, using sigma0)")
         sigma_y_eff = sigma0_pa
    else:
         sigma_y_eff = sigma0_pa + ky_sqrtm / np.sqrt(dG) # Hall-Petch

    term1 = (2.0 / 3.0) * sigma_y_eff
    log_term_arg = (3.0 * sigma_y_eff) / (2.0 * E_pa)

    if log_term_arg <= 1e-12: # Use tolerance instead of direct <= 0 check for log
         log_term = np.log(1e-12) # Avoid log(0) or log(negative) issues
         logging.warning(f"Log term argument for Ry is near/below zero ({log_term_arg:.2e}). Clamping log term.")
    else:
         log_term = np.log(log_term_arg)

    Ry = term1 * (1.0 - log_term)
    # Ensure Ry is physically reasonable (non-negative pressure unlikely needed here, but check)
    # if Ry < 0:
    #     logging.warning(f"Calculated Ry is negative ({Ry:.2e} Pa). Check parameters.")
    #     return 0 # Or handle as appropriate
    return Ry


def _N1_paper(dG, N0_GB_per_m3, d0_G_m):
    """ Calculates N1 (grain boundary nucleation site density) using Eq. (8) """
    # N1 [m^-3] = N0_GB [m^-3] * d0_G [m] / dG [m]
    if np.isclose(dG, 0) or dG < 0 or np.isinf(dG):
        return 0.0 # No grain boundaries for single crystal (infinite dG) or invalid dG
    else:
        return N0_GB_per_m3 * (d0_G_m / dG)

def _zeta_alpha_paper(beta_alpha):
    """ Calculates zeta_alpha factor based on definition below Eq. (9) """
    # zeta_alpha = 2^beta * (beta-1) * Gamma(beta) * Product[1/(9+2i)]_{i=1 to beta}
    # Paper specifies beta=3 or beta=10
    if not isinstance(beta_alpha, int) or beta_alpha < 1:
        logging.warning(f"zeta_alpha calculation requires integer beta_alpha >= 1. Got {beta_alpha}. Returning NaN.")
        return np.nan
    if beta_alpha == 1:
        return 1.0 # As stated below Eq. (9)

    try:
        product_val = 1.0
        # The product is finite, up to floor(beta_alpha), which is just beta_alpha since it's integer
        for i in range(1, beta_alpha + 1):
            term = (9.0 + 2.0 * i)
            if np.isclose(term, 0): return np.inf
            product_val /= term

        gamma_val = gamma_func(beta_alpha)
        prefactor = (2.0**beta_alpha) * (beta_alpha - 1.0)
        zeta = prefactor * gamma_val * product_val
        return zeta

    except Exception as e:
        logging.error(f"Error calculating zeta_alpha for beta={beta_alpha}: {e}")
        return np.nan

def calculate_expansion_rate_PRL(Sigma_m_star_Pa, dG, params):
    """
    Calculates the expansion rate (v_dot/v_0) based on Eq. (10) from
    Wilkerson & Ramesh, PRL 117, 215503 (2016).

    Args:
        Sigma_m_star_Pa (float or np.array): Spall strength value(s) in Pascals.
        dG (float): Grain size in meters (use np.inf for single crystal).
        params (dict): Dictionary containing Wilkerson model parameters from the paper
                       (sigma0_pa, ky_sqrtm, E_pa, Reos_pa, K0_pa, rho, N2, N0_GB, d0_G).

    Returns:
        float or np.array: Corresponding expansion rate(s) (v_dot/v_0) in s^-1.
                           Returns NaN if calculation fails or inputs are invalid.
    """
    # Ensure input is array for vectorized operations
    Sigma_m_star_Pa = np.asarray(Sigma_m_star_Pa)

    # Extract parameters (ensure they are in SI units)
    try:
        sigma0_pa = params['sigma0_pa']
        ky_sqrtm = params['ky_sqrtm']
        E_pa = params['E_pa']
        Reos_pa = params['Reos_pa']
        K0_pa = params['K0_pa']
        rho = params['rho']
        N2_per_m3 = params['N2']         # Grain interior site density [m^-3]
        N0_GB_per_m3 = params['N0_GB']  # Ref GB site density [m^-3]
        d0_G_m = params['d0_G']           # Ref grain size [m]
    except KeyError as ke:
        logging.error(f"Missing key in params dictionary for PRL model: {ke}")
        return np.full_like(Sigma_m_star_Pa, np.nan, dtype=float)

    # --- Constants and intermediate calculations ---
    KAPPA_HAT = (4.0/3.0) * math.pi * (8.0/33.0)**(3.0/2.0)
    try:
        CB_m_s = math.sqrt(K0_pa / rho) # Bulk wave speed
    except ValueError:
        logging.error("Invalid K0_pa or rho for CB calculation.")
        return np.full_like(Sigma_m_star_Pa, np.nan, dtype=float)

    # Calculate Ry (cavitation initiation pressure) - calculated once for the given dG
    Ry_pa = _Ry_paper(sigma0_pa, ky_sqrtm, E_pa, dG)
    if pd.isna(Ry_pa):
        logging.error("Failed to calculate Ry.")
        return np.full_like(Sigma_m_star_Pa, np.nan, dtype=float)

    # --- Define the two families (alpha=1: GB, alpha=2: Interior) ---
    families = {
        1: {'beta': 3, 'N': _N1_paper(dG, N0_GB_per_m3, d0_G_m)}, # GB sites
        2: {'beta': 10, 'N': N2_per_m3}                           # Interior sites
    }

    # --- Calculate Summation Term in Eq. (10) ---
    total_sum_term = np.zeros_like(Sigma_m_star_Pa, dtype=float)

    # Check Reos vs Ry - denominator must be positive
    Reos_minus_Ry = Reos_pa - Ry_pa
    if Reos_minus_Ry <= 0:
        logging.warning(f"Reos_pa ({Reos_pa:.2e}) is not greater than Ry_pa ({Ry_pa:.2e}). Result invalid.")
        return np.full_like(Sigma_m_star_Pa, np.nan, dtype=float)

    # Macaulay bracket for the spall strength difference term
    diff_term = np.maximum(0.0, Sigma_m_star_Pa - Ry_pa)

    for alpha, props in families.items():
        beta_alpha = props['beta']
        N_alpha = props['N'] # Site density for this family [m^-3]

        if N_alpha <= 0: # Skip family if density is zero (e.g., N1 for single crystal)
            continue

        zeta_alpha = _zeta_alpha_paper(beta_alpha)
        if pd.isna(zeta_alpha):
            logging.error(f"Failed to calculate zeta_alpha for alpha={alpha}, beta={beta_alpha}")
            return np.full_like(Sigma_m_star_Pa, np.nan, dtype=float)

        zeta_hat_alpha = (beta_alpha + 9.0/2.0) * zeta_alpha
        c_star_hat_alpha = (9.0 + 2.0 * beta_alpha) / (7.0 + 2.0 * beta_alpha)

        denominator = Reos_minus_Ry**beta_alpha

        # Calculate numerator term = [<c_star*(Sigma_m* - Ry)>]^(beta + 7/2)
        inner_macaulay = np.maximum(0.0, c_star_hat_alpha * diff_term)
        numerator = inner_macaulay**(beta_alpha + 7.0/2.0)

        term_alpha = (zeta_hat_alpha * N_alpha * numerator) / denominator
        total_sum_term += term_alpha

    # --- Calculate Expansion Rate using Eq. (10) ---
    factor = (KAPPA_HAT * CB_m_s**3) / (K0_pa**(7.0/2.0))
    v_dot_over_v0_cubed = factor * total_sum_term

    # Handle potential negative results before taking cube root (shouldn't happen if inputs are physical)
    v_dot_over_v0_cubed = np.maximum(0.0, v_dot_over_v0_cubed)
    v_dot_over_v0 = np.cbrt(v_dot_over_v0_cubed)

    return v_dot_over_v0
#=============================================================================#