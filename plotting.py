# spade_analysis/plotting.py
"""
Plotting functions for visualizing analysis results and comparisons.
(Moved models import into functions to fix circular import)
(Consolidated imports)
(Added explicit save logging and failure reporting)
(Commented out 3D scatter plot for stability)
(Commented out old Wilkerson model plotting calls)
(Integrated PRL model calculation and plotting)
(Added robustness and improved logging)
"""
# --- Consolidated Imports ---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import warnings
import logging
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler # Needed for combined plot scaling check
from scipy.interpolate import interp1d # Needed for combined trace plot
import math # Needed for PRL model helpers
from scipy.special import gamma as gamma_func # Needed for PRL model helpers


from . import utils
from .literature import load_literature_data
# --- End Consolidated Imports ---

# --- Plotting Style Configuration ---
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    logging.warning("Seaborn-v0_8-darkgrid style not available. Using default style.")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 7
plt.rcParams['errorbar.capsize'] = 3
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# --- Plotting Functions ---

def plot_velocity_comparison(input_directory, file_pattern, output_filename,
                             title="Velocity Trace Comparison", time_unit='ns'):
    """
    Plots multiple velocity traces from CSV files matching a pattern onto a single graph.

    Args:
        input_directory (str): Directory containing the CSV files.
        file_pattern (str): Glob pattern to match files (e.g., '*.csv').
        output_filename (str): Path to save the output plot image.
        title (str, optional): Title for the plot. Defaults to "Velocity Trace Comparison".
        time_unit (str, optional): Unit for the time axis ('ns' or 'us'). Defaults to 'ns'.
    """
    fig, ax = plt.subplots()
    files = sorted(glob.glob(os.path.join(input_directory, file_pattern)))
    plotted_files = 0
    logging.info(f"Generating velocity comparison plot: {output_filename}")
    logging.info(f" Found {len(files)} files matching pattern.")

    for file in files:
        try:
            df = pd.read_csv(file)
            # Determine time and velocity columns (handle variations)
            time_col = next((col for col in df.columns if 'Time' in col), None)
            velocity_col = next((col for col in df.columns if 'Velocity' in col or 'Speed' in col), None)

            if time_col and velocity_col:
                time_data = df[time_col]
                velocity_data = df[velocity_col]
                # Handle time unit conversion
                if time_unit == 'us' and 'ns' in time_col:
                    time_data = time_data / 1000.0
                elif time_unit == 'ns' and 'us' in time_col:
                     time_data = time_data * 1000.0

                label = os.path.basename(file).replace('.csv', '') # Simple label from filename
                ax.plot(time_data, velocity_data, label=label)
                plotted_files += 1
            else:
                logging.warning(f"Could not find time/velocity columns in {file}. Skipping.")
        except Exception as e:
            logging.error(f"Error processing file {file} for velocity comparison plot: {e}")

    xlabel = f'Time ({time_unit})'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(title)
    if plotted_files > 0:
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
    else:
         logging.warning("No files were successfully plotted.")

    # --- Save Plot ---
    plot_saved = False
    try:
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150)
        logging.info(f"Successfully saved velocity comparison plot to: {output_filename}")
        plot_saved = True
    except Exception as e:
        logging.exception(f"Error saving velocity comparison plot {output_filename}: {e}")
    finally:
        plt.close(fig)
        if not plot_saved:
            logging.error(f"Velocity comparison plot generation failed or was skipped, plot NOT saved: {output_filename}")


def plot_spall_vs_strain_rate(df, output_filename, title="Spall Strength vs. Strain Rate",
                               xlim=None, ylim=None, log_scale=True, show_legend=True,
                               color_map=None, marker_map=None,
                               spall_col='Spall Strength (GPa)', strain_rate_col='Strain Rate (s^-1)',
                               material_col='Material', literature_data_file=None,
                               spall_unc_col=None, filter_high_error_perc=None, group_by_material=True):
    """ Plots Spall Strength vs. Strain Rate for different materials, optionally with literature data. """
    if color_map is None: color_map = utils.COLOR_MAPPING
    if marker_map is None: marker_map = utils.MARKER_MAPPING

    fig, ax = plt.subplots()
    plotted_exp_materials = set()
    exp_handles_proxies = {}
    lit_handles = {}
    exp_data_plotted = False
    lit_data_plotted = False
    plot_saved = False
    logging.info(f"Generating spall vs strain rate plot: {output_filename}")

    try:
        # --- Plot Experimental Data ---
        logging.info(f"Processing experimental data from DataFrame for Strain Rate plot...")
        if df is not None and not df.empty:
            data = df.copy()
            required_exp_cols = [strain_rate_col, spall_col, material_col]
            if not all(col in data.columns for col in required_exp_cols):
                logging.warning(f"Skipping experimental data plotting. DataFrame missing required columns: {', '.join(c for c in required_exp_cols if c not in data.columns)}")
            else:
                x_data = pd.to_numeric(data[strain_rate_col], errors='coerce')
                y_data = pd.to_numeric(data[spall_col], errors='coerce')
                y_err = pd.to_numeric(data[spall_unc_col].abs(), errors='coerce') if spall_unc_col and spall_unc_col in data.columns else None
                plot_df = pd.DataFrame({'x': x_data, 'y': y_data, material_col: data.get(material_col)})
                if y_err is not None: plot_df['y_err'] = y_err
                initial_rows = len(plot_df)
                plot_df.dropna(subset=['x', 'y', material_col], inplace=True)
                if log_scale: plot_df = plot_df[plot_df['x'] > 0] # Ensure positive for log scale
                if len(plot_df) < initial_rows: logging.warning(f"Removed {initial_rows - len(plot_df)} rows with NaN/non-positive values from experimental DataFrame.")
                logging.debug(f"Plotting DF shape after NaN/filter drop: {plot_df.shape}")

                if not plot_df.empty:
                    if filter_high_error_perc is not None and 'y_err' in plot_df.columns:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            relative_error = np.abs(plot_df['y_err'] / plot_df['y'].replace(0, np.nan)) * 100
                        valid_points_mask = relative_error <= filter_high_error_perc
                        valid_points_mask = valid_points_mask | relative_error.isna() # Keep points where error couldn't be calculated
                        if not valid_points_mask.all():
                            removed_count = sum(~valid_points_mask)
                            logging.info(f"  Filtering experimental data: Removing {removed_count} points with >{filter_high_error_perc:.0f}% relative Y error.")
                            plot_df = plot_df[valid_points_mask]
                        logging.debug(f"Plotting DF shape after error filter: {plot_df.shape}")

                    if not plot_df.empty:
                        exp_data_plotted = True
                        logging.debug(f"Plotting experimental data grouped by '{material_col}' column...")
                        for material_label, group_data in plot_df.groupby(material_col):
                            logging.debug(f"  Plotting group: '{material_label}' ({len(group_data)} points)")
                            color = color_map.get(material_label, color_map.get("Default", "gray"))
                            marker = marker_map.get(material_label, marker_map.get("Default", "x"))
                            logging.debug(f"    -> Label='{material_label}', Color='{color}', Marker='{marker}'")
                            ax.errorbar(group_data['x'], group_data['y'],
                                        yerr=group_data.get('y_err'), xerr=None,
                                        fmt=marker, color=color, ecolor='darkgray', elinewidth=1,
                                        capsize=plt.rcParams['errorbar.capsize'], alpha=0.8,
                                        label='_nolegend_', markersize=plt.rcParams['lines.markersize'])
                            if group_by_material and material_label not in plotted_exp_materials:
                                plotted_exp_materials.add(material_label)
                                proxy = plt.Line2D([0], [0], linestyle='None', marker=marker, color=color, markersize=plt.rcParams['lines.markersize'])
                                exp_handles_proxies[material_label] = proxy
                    else: logging.warning("No valid experimental data points remain after filtering.")
                else: logging.warning("No valid experimental data points remain after NaN removal.")
        else: logging.warning("Experimental data DataFrame is None or empty.")

        # --- Plot Literature Data ---
        if literature_data_file:
            logging.info(f"Attempting to load literature data from: {literature_data_file}")
            try:
                # Ensure required columns for this plot type are checked
                lit_req_cols = [strain_rate_col, spall_col, 'Source']
                lit_data_raw = load_literature_data(literature_data_file, required_columns=lit_req_cols)
                if lit_data_raw is not None and not lit_data_raw.empty:
                    lit_data = lit_data_raw[lit_req_cols].copy()
                    lit_data[strain_rate_col] = pd.to_numeric(lit_data[strain_rate_col], errors='coerce')
                    lit_data[spall_col] = pd.to_numeric(lit_data[spall_col], errors='coerce')
                    lit_data.dropna(inplace=True)
                    if log_scale: lit_data = lit_data[lit_data[strain_rate_col] > 0]
                    logging.debug(f"Literature DF shape after NaN/filter drop: {lit_data.shape}")

                    if not lit_data.empty:
                         lit_data_plotted = True
                         sources = lit_data['Source'].unique()
                         prop_cycle = plt.rcParams['axes.prop_cycle']
                         lit_colors = prop_cycle.by_key()['color']
                         logging.info(f"Plotting literature data for sources: {', '.join(sources)}")
                         for idx, source in enumerate(sources):
                            source_data = lit_data[lit_data['Source'] == source]
                            if source_data.empty: continue
                            logging.debug(f"  Plotting literature source: {source} ({len(source_data)} points)")
                            lit_x = source_data[strain_rate_col]
                            lit_y = source_data[spall_col]
                            lit_marker = marker_map.get(source, marker_map.get("Default", "d"))
                            # Offset color cycle index to avoid clashing with exp data
                            color_cycle_idx = (len(plotted_exp_materials) + idx) % len(lit_colors)
                            lit_color = color_map.get(source, lit_colors[color_cycle_idx])
                            handle = ax.scatter(lit_x, lit_y, s=80, alpha=0.7, color=lit_color, label=source, marker=lit_marker, edgecolors='grey', zorder=4)
                            lit_handles[source] = handle
                    else: logging.warning("No valid data points in literature file after NaN/filter removal.")
                # No need for else here, load_literature_data logs errors
            except Exception as e:
                logging.exception(f"Error reading or processing literature file {literature_data_file}: {e}")

        # --- Customize Plot ---
        logging.debug("Customizing plot axes and labels...")
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(strain_rate_col, fontsize=20)
        ax.set_ylabel(spall_col, fontsize=20)
        if log_scale:
            ax.set_xscale('log')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major')
        ax.grid(True, which='both' if log_scale else 'major', linestyle='--', alpha=0.6)

        # --- Create Combined Legend ---
        logging.debug("Creating legend...")
        legend_handles = []
        legend_labels = []
        if group_by_material:
            sorted_materials = sorted(exp_handles_proxies.keys())
            for material_label in sorted_materials:
                proxy_handle = exp_handles_proxies[material_label]
                legend_handles.append(proxy_handle)
                legend_labels.append(f"{material_label} Exp.")
        sorted_lit_sources = sorted(lit_handles.keys())
        for source in sorted_lit_sources:
            handle = lit_handles[source]
            legend_handles.append(handle)
            legend_labels.append(source)
        if show_legend and legend_handles:
            ax.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize=14)
        elif exp_data_plotted or lit_data_plotted:
            logging.warning("Data was plotted, but legend could not be created (check handle generation).")
        else:
            logging.info("No experimental or literature data plotted, legend skipped.")

        try:
            plt.tight_layout()
        except Exception as tle:
            logging.warning(f"tight_layout failed for {output_filename}: {tle}")

        # --- Save Plot ---
        logging.info(f"Attempting to save plot: {output_filename}")
        plt.savefig(output_filename, dpi=150)
        logging.info(f"Successfully saved spall vs strain rate plot to: {output_filename}")
        plot_saved = True

    except Exception as e:
        logging.exception(f"Error occurred during generation of plot {output_filename}: {e}")
    finally:
        logging.debug(f"Closing figure for {output_filename}")
        plt.close(fig)
        if not plot_saved:
             logging.error(f"Plot generation failed or was skipped, plot NOT saved: {output_filename}")


def plot_spall_vs_shock_stress(df, output_filename, title="Spall Strength vs. Shock Stress",
                               xlim=None, ylim=None, show_legend=True,
                               color_map=None, marker_map=None,
                               spall_col='Spall Strength (GPa)', shock_stress_col='Peak Shock Pressure (GPa)',
                               material_col='Material', literature_data_file=None,
                               spall_unc_col=None, filter_high_error_perc=None, group_by_material=True,
                               lit_shock_col='Shock Stress (GPa)', lit_spall_col='Spall Strength (GPa)', lit_source_col='Source'):
    """ Plots Spall Strength vs. Shock Stress for different materials, optionally with literature data. """
    if color_map is None: color_map = utils.COLOR_MAPPING
    if marker_map is None: marker_map = utils.MARKER_MAPPING

    fig, ax = plt.subplots()
    plotted_exp_materials = set()
    exp_handles_proxies = {}
    lit_handles = {}
    exp_data_plotted = False
    lit_data_plotted = False
    plot_saved = False
    logging.info(f"Generating spall vs shock stress plot: {output_filename}")

    try:
        # --- Plot Experimental Data ---
        logging.info(f"Processing experimental data from DataFrame for Shock Stress plot...")
        if df is not None and not df.empty:
            data = df.copy()
            required_exp_cols = [shock_stress_col, spall_col, material_col]
            if not all(col in data.columns for col in required_exp_cols):
                logging.warning(f"Skipping experimental data plotting. DataFrame missing required columns: {', '.join(c for c in required_exp_cols if c not in data.columns)}")
            else:
                x_data = pd.to_numeric(data[shock_stress_col], errors='coerce')
                y_data = pd.to_numeric(data[spall_col], errors='coerce')
                y_err = pd.to_numeric(data[spall_unc_col].abs(), errors='coerce') if spall_unc_col and spall_unc_col in data.columns else None
                plot_df = pd.DataFrame({'x': x_data, 'y': y_data, material_col: data.get(material_col)})
                if y_err is not None: plot_df['y_err'] = y_err
                initial_rows = len(plot_df)
                plot_df.dropna(subset=['x', 'y', material_col], inplace=True)
                if len(plot_df) < initial_rows: logging.warning(f"Removed {initial_rows - len(plot_df)} rows with NaN values from experimental DataFrame.")
                logging.debug(f"Plotting DF shape after NaN drop: {plot_df.shape}")

                if not plot_df.empty:
                    if filter_high_error_perc is not None and 'y_err' in plot_df.columns:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            relative_error = np.abs(plot_df['y_err'] / plot_df['y'].replace(0, np.nan)) * 100
                        valid_points_mask = relative_error <= filter_high_error_perc
                        valid_points_mask = valid_points_mask | relative_error.isna() # Keep points where error couldn't be calculated
                        if not valid_points_mask.all():
                            removed_count = sum(~valid_points_mask)
                            logging.info(f"  Filtering experimental data: Removing {removed_count} points with >{filter_high_error_perc:.0f}% relative Spall error.")
                            plot_df = plot_df[valid_points_mask]
                        logging.debug(f"Plotting DF shape after error filter: {plot_df.shape}")

                    if not plot_df.empty:
                        exp_data_plotted = True
                        logging.debug(f"Plotting experimental data grouped by '{material_col}' column...")
                        for material_label, group_data in plot_df.groupby(material_col):
                            logging.debug(f"  Plotting group: '{material_label}' ({len(group_data)} points)")
                            color = color_map.get(material_label, color_map.get("Default", "gray"))
                            marker = marker_map.get(material_label, marker_map.get("Default", "x"))
                            logging.debug(f"    -> Label='{material_label}', Color='{color}', Marker='{marker}'")
                            ax.errorbar(group_data['x'], group_data['y'], yerr=group_data.get('y_err'), xerr=None,
                                        fmt=marker, color=color, ecolor='darkgray', elinewidth=1,
                                        capsize=plt.rcParams['errorbar.capsize'], alpha=0.8,
                                        label='_nolegend_', markersize=plt.rcParams['lines.markersize'])
                            if group_by_material and material_label not in plotted_exp_materials:
                                plotted_exp_materials.add(material_label)
                                proxy = plt.Line2D([0], [0], linestyle='None', marker=marker, color=color, markersize=plt.rcParams['lines.markersize'])
                                exp_handles_proxies[material_label] = proxy
                    else: logging.warning("No valid experimental data points remain after filtering.")
                else: logging.warning("No valid experimental data points remain after NaN removal.")
        else: logging.warning("Experimental data DataFrame is None or empty.")

        # --- Plot Literature Data ---
        if literature_data_file:
            logging.info(f"Attempting to load literature data from: {literature_data_file}")
            try:
                lit_req_cols = [lit_shock_col, lit_spall_col, lit_source_col]
                lit_data_raw = load_literature_data(literature_data_file, required_columns=lit_req_cols)
                if lit_data_raw is not None and not lit_data_raw.empty:
                    lit_data = lit_data_raw[lit_req_cols].copy()
                    lit_data[lit_shock_col] = pd.to_numeric(lit_data[lit_shock_col], errors='coerce')
                    lit_data[lit_spall_col] = pd.to_numeric(lit_data[lit_spall_col], errors='coerce')
                    lit_data.dropna(subset=[lit_shock_col, lit_spall_col], inplace=True)
                    logging.debug(f"Literature DF shape after NaN drop: {lit_data.shape}")

                    if not lit_data.empty:
                         lit_data_plotted = True
                         sources = lit_data[lit_source_col].unique()
                         prop_cycle = plt.rcParams['axes.prop_cycle']
                         lit_colors = prop_cycle.by_key()['color']
                         logging.info(f"Plotting literature data for sources: {', '.join(sources)}")
                         for idx, source in enumerate(sources):
                            source_data = lit_data[lit_data[lit_source_col] == source]
                            if source_data.empty: continue
                            logging.debug(f"  Plotting literature source: {source} ({len(source_data)} points)")
                            lit_x = source_data[lit_shock_col]
                            lit_y = source_data[lit_spall_col]
                            lit_marker = marker_map.get(source, marker_map.get("Default", "d"))
                            # Offset color cycle index
                            color_cycle_idx = (len(plotted_exp_materials) + idx) % len(lit_colors)
                            lit_color = color_map.get(source, lit_colors[color_cycle_idx])
                            handle = ax.scatter(lit_x, lit_y, s=80, alpha=0.7, color=lit_color, label=source, marker=lit_marker, edgecolors='grey', zorder=4)
                            lit_handles[source] = handle
                    else: logging.warning("No valid data points in literature file after NaN removal.")
                # No need for else here, load_literature_data logs errors
            except Exception as e:
                logging.exception(f"Error reading or processing literature file {literature_data_file}: {e}")

        # --- Customize Plot ---
        logging.debug("Customizing plot axes and labels...")
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(shock_stress_col, fontsize=20)
        ax.set_ylabel(spall_col, fontsize=20)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major')
        ax.grid(True, which='major', linestyle='--', alpha=0.6)

        # --- Create Combined Legend ---
        logging.debug("Creating legend...")
        legend_handles = []
        legend_labels = []
        if group_by_material:
            sorted_materials = sorted(exp_handles_proxies.keys())
            for material_label in sorted_materials:
                proxy_handle = exp_handles_proxies[material_label]
                legend_handles.append(proxy_handle)
                legend_labels.append(f"{material_label} Exp.")
        sorted_lit_sources = sorted(lit_handles.keys())
        for source in sorted_lit_sources:
            handle = lit_handles[source]
            legend_handles.append(handle)
            legend_labels.append(source)
        if show_legend and legend_handles:
            ax.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize=14)
        elif exp_data_plotted or lit_data_plotted:
            logging.warning("Data was plotted, but legend could not be created (check handle generation).")
        else:
            logging.info("No experimental or literature data plotted, legend skipped.")

        try:
            plt.tight_layout()
        except Exception as tle:
            logging.warning(f"tight_layout failed for {output_filename}: {tle}")

        # --- Save Plot ---
        logging.info(f"Attempting to save plot: {output_filename}")
        plt.savefig(output_filename, dpi=150)
        logging.info(f"Successfully saved spall vs shock stress plot to: {output_filename}")
        plot_saved = True

    except Exception as e:
        logging.exception(f"Error occurred during generation of plot {output_filename}: {e}")
    finally:
        logging.debug(f"Closing figure for {output_filename}")
        plt.close(fig)
        if not plot_saved:
             logging.error(f"Plot generation failed or was skipped, plot NOT saved: {output_filename}")


def plot_wilkerson_comparison(
    experimental_data_df,
    output_filename,
    wilkerson_params,
    # Note: strain_rate_col is used for plotting experimental data,
    #       but the PRL model is plotted against expansion rate.
    strain_rate_col='Strain Rate (s^-1)',
    spall_col='Spall Strength (GPa)',
    spall_unc_col=None,
    material_col='Material',
    title="Spall Strength vs. Expansion Rate (PRL Model Comparison)",
    xlabel='Expansion Rate ($\dot{v}/v_0$) [s$^{-1}$]', # Updated label
    ylabel='Spall Strength (GPa)',
    xlim=None, ylim=None, log_scale=True, show_legend=True, # xlim now for expansion rate
    color_map=None, marker_map=None,
    filter_high_error_perc=100,
    literature_file=None):
    """
    Plots experimental data (Spall vs Strain Rate) against the Wilkerson PRL model prediction.
    The PRL model curve itself is plotted against Expansion Rate.

    Args:
        experimental_data_df (pd.DataFrame): DataFrame containing experimental results.
        output_filename (str): Path to save the plot.
        wilkerson_params (dict): Dictionary of parameters for the Wilkerson PRL model.
                                 Must include 'dG' (grain size in meters) for the model calculation.
        strain_rate_col (str): Name of the strain rate column in df (for plotting exp data).
        spall_col (str): Name of the spall strength column in df.
        spall_unc_col (str, optional): Name of spall strength uncertainty column.
        material_col (str): Name of the material identifier column in df.
        title (str): Plot title.
        xlabel (str): X-axis label (should be Expansion Rate).
        ylabel (str): Y-axis label.
        xlim (tuple, optional): X-axis limits (min, max) for Expansion Rate.
        ylim (tuple, optional): Y-axis limits (min, max).
        log_scale (bool): Whether to use log scale for the x-axis (Expansion Rate).
        show_legend (bool): Whether to display the legend.
        color_map (dict, optional): Mapping from material label to color.
        marker_map (dict, optional): Mapping from material label to marker style.
        filter_high_error_perc (int, optional): Filter exp data with Y error > this percentage.
        literature_file (str, optional): Path to literature data CSV file (plotted vs Strain Rate).
    """
    # *** Import model function here to avoid potential circular imports ***
    from .models import calculate_expansion_rate_PRL # Import the NEW PRL model function

    if color_map is None: color_map = utils.COLOR_MAPPING
    if marker_map is None: marker_map = utils.MARKER_MAPPING

    fig, ax = plt.subplots()
    plotted_elements = {} # Store handles for legend {label: handle}
    plotted_exp_materials = []
    model_line_plotted = False
    plot_saved = False
    logging.info(f"Generating Wilkerson PRL comparison plot: {output_filename}")
    logging.debug(f" Using Wilkerson parameters: {wilkerson_params}")

    try:
        # --- Plot Experimental Data (vs Strain Rate) ---
        logging.debug("Plotting experimental data (vs Strain Rate)...")
        # Ensure required columns for experimental data exist
        required_exp_cols = [strain_rate_col, spall_col, material_col]
        if not all(col in experimental_data_df.columns for col in required_exp_cols):
            logging.warning(f"Skipping experimental data plotting. DataFrame missing required columns: {', '.join(c for c in required_exp_cols if c not in experimental_data_df.columns)}")
            plot_df = pd.DataFrame() # Empty df
        else:
            plot_df = experimental_data_df[required_exp_cols].dropna().copy()
            # Ensure positive strain rate if log scale is used for exp data axis
            if log_scale:
                 plot_df = plot_df[pd.to_numeric(plot_df[strain_rate_col], errors='coerce') > 0]

        if plot_df.empty:
            logging.warning("No valid experimental data points found for Wilkerson comparison plot.")
        else:
            # Apply error filtering if needed
            if filter_high_error_perc is not None and spall_unc_col and spall_unc_col in experimental_data_df.columns:
                 y_err = pd.to_numeric(experimental_data_df.loc[plot_df.index, spall_unc_col].abs(), errors='coerce')
                 if y_err is not None:
                     with np.errstate(divide='ignore', invalid='ignore'):
                         relative_error = np.abs(y_err / plot_df[spall_col].replace(0, np.nan)) * 100
                     valid_points_mask = relative_error <= filter_high_error_perc
                     valid_points_mask = valid_points_mask | relative_error.isna()
                     if not valid_points_mask.all():
                         removed_count = sum(~valid_points_mask)
                         logging.info(f"  Filtering experimental data: Removing {removed_count} points with >{filter_high_error_perc:.0f}% relative Y error.")
                         plot_df = plot_df[valid_points_mask]

            if not plot_df.empty:
                for material, group in plot_df.groupby(material_col):
                    material_label = utils.MATERIAL_MAPPING.get(material, material)
                    color = color_map.get(material_label, 'black')
                    marker = marker_map.get(material_label, 'o')
                    if not group.empty:
                        # Plot experimental data against STRAIN RATE
                        handle = ax.scatter(group[strain_rate_col], group[spall_col],
                                            label=f'{material_label} (Exp.)', color=color, marker=marker, s=50, alpha=0.8, zorder=5)
                        plotted_elements[f'{material_label} (Exp.)'] = handle
                        plotted_exp_materials.append(material_label)
                    else:
                         logging.debug(f"No valid experimental data for material '{material_label}' after filtering.")
                logging.info(f" Plotted experimental data for: {plotted_exp_materials}")
            else:
                logging.warning("No valid experimental data points remain after filtering.")

        # --- Plot Literature Data (vs Strain Rate) ---
        if literature_file:
            logging.debug(f"Attempting to plot literature data from: {literature_file}")
            # Ensure required columns for literature plotting exist
            lit_req_cols = [strain_rate_col, spall_col, 'Source']
            lit_df = load_literature_data(literature_file, required_columns=lit_req_cols)
            if lit_df is not None:
                plot_lit_df = lit_df.dropna(subset=[spall_col, strain_rate_col]).copy()
                if log_scale: plot_lit_df = plot_lit_df[pd.to_numeric(plot_lit_df[strain_rate_col], errors='coerce') > 0]

                if plot_lit_df.empty:
                     logging.warning("No valid literature data points found after filtering.")
                else:
                    for source, group in plot_lit_df.groupby('Source'):
                        color = color_map.get(source, 'grey')
                        marker = marker_map.get(source, 'x')

                        if not group.empty:
                             # Plot literature data against STRAIN RATE
                             handle = ax.scatter(group[strain_rate_col], group[spall_col],
                                                label=f'{source} (Lit.)', color=color, marker=marker, s=40, alpha=0.6, zorder=4)
                             plotted_elements[f'{source} (Lit.)'] = handle
                        else:
                             logging.debug(f"No valid literature data for source '{source}'")
                    logging.info(" Plotted literature data.")
            else:
                logging.warning(f"Could not load or process literature data from {literature_file}.")

        # --- Calculate and Plot Wilkerson Model (PRL Version) ---
        logging.info("Calculating Wilkerson model predictions (PRL paper)...")
        model_line = None # Initialize
        try:
            required_wp_keys = ['sigma0_pa', 'ky_sqrtm', 'E_pa', 'Reos_pa', 'K0_pa', 'rho', 'N2', 'N0_GB', 'd0_G']
            if not all(key in wilkerson_params for key in required_wp_keys):
                missing_keys = [key for key in required_wp_keys if key not in wilkerson_params]
                logging.error(f"Missing required Wilkerson parameters for PRL model: {missing_keys}. Skipping model line.")
            else:
                # Generate a range of SPALL STRENGTHS (y-axis)
                min_spall_gpa = max(0.1, ylim[0] if ylim and ylim[0] > 0 else 0.1)
                max_spall_gpa = ylim[1] if ylim and ylim[1] > min_spall_gpa else 25
                num_points = 200
                model_spall_strengths_gpa = np.linspace(min_spall_gpa, max_spall_gpa, num_points)
                model_spall_strengths_pa = model_spall_strengths_gpa * 1e9

                # Get dG for this specific plot
                dG_val = wilkerson_params.get('dG')
                if dG_val is None:
                     logging.error("Grain size 'dG' not found in wilkerson_params. Cannot calculate model line.")
                else:
                     # Call the NEW function
                     dG_input = dG_val
                     if dG_val <= 0: dG_input = np.inf # Use np.inf for single crystal representation

                     expansion_rates = calculate_expansion_rate_PRL(
                         Sigma_m_star_Pa=model_spall_strengths_pa,
                         dG=dG_input,
                         params=wilkerson_params
                     )

                     # Filter out NaNs or non-positive rates before plotting
                     valid_mask = pd.notna(expansion_rates) & (expansion_rates > 1e-9) # Check > 0 for log plot
                     plot_expansion_rates = expansion_rates[valid_mask]
                     plot_spall_strengths_gpa = model_spall_strengths_gpa[valid_mask]

                     if len(plot_expansion_rates) > 0:
                         logging.debug("Plotting Wilkerson model line (PRL)...")
                         logging.debug(f"      Data points (first 5): X(ExpRate)={plot_expansion_rates[:5]}, Y(SpallGPa)={plot_spall_strengths_gpa[:5]}")
                         # Plot EXPANSION RATE (x) vs SPALL STRENGTH (y)
                         model_label = f'Wilkerson Model (PRL, dG={dG_val*1e6:.1f} um)' if dG_val > 0 and np.isfinite(dG_val) else 'Wilkerson Model (PRL, SC)'
                         model_line, = ax.plot(plot_expansion_rates,
                                               plot_spall_strengths_gpa,
                                               color='black', linestyle='--', linewidth=2, label=model_label, zorder=10)
                         plotted_elements[model_label] = model_line
                         model_line_plotted = True
                         logging.info(f" Plotted Wilkerson model line (PRL) for dG={dG_val}.")
                     else:
                         logging.warning(f" Wilkerson model calculation (PRL) resulted in no valid points for dG={dG_val}. Line not plotted.")

        except Exception as model_calc_e:
            logging.exception(f"Error during Wilkerson model calculation (PRL): {model_calc_e}. Model line not plotted.")

        # --- *** COMMENTED OUT OLD WILKERSON MODEL BLOCK *** ---
        # logging.info("Calculating Complex Wilkerson model predictions...")
        # try:
        #     # ... (code for old model calculation and plotting) ...
        # except Exception as old_model_e:
        #      logging.exception(f"Error during OLD Wilkerson model calculation: {old_model_e}")
        # --- *** END OF COMMENTED OUT BLOCK *** ---

        # --- Customize Plot ---
        logging.debug("Customizing plot axes and labels...")
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel) # Should be 'Expansion Rate...'
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_xscale('log')
        if xlim: ax.set_xlim(xlim) # xlim for Expansion Rate
        if ylim: ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major')
        ax.grid(True, which='both' if log_scale else 'major', linestyle='--', alpha=0.6)

        # Create legend from collected handles/labels
        if show_legend and plotted_elements:
            # Sort labels for consistent legend order (optional)
            sorted_labels = sorted(plotted_elements.keys())
            sorted_handles = [plotted_elements[lbl] for lbl in sorted_labels]
            ax.legend(handles=sorted_handles, labels=sorted_labels, loc='best', fontsize=14)

        # --- Save Plot ---
        try:
            plt.tight_layout()
            plt.savefig(output_filename, dpi=150)
            logging.info(f"Successfully saved Wilkerson comparison plot to: {output_filename}")
            plot_saved = True
        except Exception as e:
            logging.exception(f"Error saving Wilkerson comparison plot {output_filename}: {e}")

    except Exception as e:
        logging.exception(f"Error occurred during generation of plot {output_filename}: {e}")
    finally:
        logging.debug(f"Closing figure for {output_filename}")
        plt.close(fig)
        if not plot_saved:
             logging.error(f"Plot generation failed or was skipped, plot NOT saved: {output_filename}")


def plot_spall_vs_strain_rate_multi_wilkerson(
    experimental_data_df,
    output_filename,
    literature_data_file,
    wilkerson_params_base,
    grain_sizes_dict,
    strain_rate_col='Strain Rate (s^-1)',
    spall_col='Spall Strength (GPa)',
    spall_unc_col=None,
    material_col='Material',
    title="Spall Strength vs. Expansion Rate - Multi-Wilkerson (PRL)",
    xlabel='Expansion Rate ($\dot{v}/v_0$) [s$^{-1}$]', # Updated label
    ylabel='Spall Strength (GPa)',
    log_scale=True, xlim=None, ylim=None, # xlim for expansion rate
    color_map=None, marker_map=None, model_linestyle_map = None,
    filter_high_error_perc=None,
    group_by_material=True,
    show_legend=True # Added show_legend argument
    ):
    """
    Plots experimental Spall Strength vs. Strain Rate data, literature data (vs Strain Rate),
    and overlays Wilkerson PRL model predictions (vs Expansion Rate) for specified grain sizes.

    Args:
        experimental_data_df (pd.DataFrame): DataFrame containing experimental results.
        output_filename (str): Path to save the plot.
        literature_data_file (str): Path to literature data CSV file.
        wilkerson_params_base (dict): Base dictionary of parameters for the Wilkerson PRL model.
        grain_sizes_dict (dict): Dictionary mapping labels (e.g., 'Poly (4um)') to grain sizes (dG in meters).
        strain_rate_col (str): Name of the strain rate column in df (for plotting exp/lit data).
        spall_col (str): Name of the spall strength column in df.
        spall_unc_col (str, optional): Name of spall strength uncertainty column.
        material_col (str): Name of the material identifier column in df.
        title (str): Plot title.
        xlabel (str): X-axis label (should correspond to Expansion Rate).
        ylabel (str): Y-axis label.
        xlim (tuple, optional): X-axis limits (min, max) for expansion rate.
        ylim (tuple, optional): Y-axis limits (min, max) for spall strength.
        log_scale (bool): Whether to use log scale for the x-axis.
        show_legend (bool): Whether to display the legend.
        color_map (dict, optional): Mapping from material label to color for experimental data.
        marker_map (dict, optional): Mapping from material label to marker style for experimental data.
        model_linestyle_map (dict, optional): Mapping from grain size label to linestyle for model lines.
        filter_high_error_perc (int, optional): Filter exp data with Y error > this percentage.
        group_by_material (bool): Whether to group experimental data by material for legend.
    """
    # *** Import model function here ***
    from .models import calculate_expansion_rate_PRL

    if color_map is None: color_map = utils.COLOR_MAPPING
    if marker_map is None: marker_map = utils.MARKER_MAPPING

    fig, ax = plt.subplots()
    exp_handles = {}
    lit_handles = {}
    model_handles = {}
    plotted_exp_materials = []
    model_lines_plotted = False
    plot_saved = False
    logging.info(f"Generating Multi-Wilkerson PRL comparison plot: {output_filename}")

    try:
        # --- Plot Experimental Data (vs Strain Rate) ---
        logging.debug("Plotting experimental data (vs Strain Rate)...")
        # Ensure required columns for experimental data exist
        required_exp_cols = [strain_rate_col, spall_col, material_col]
        if not all(col in experimental_data_df.columns for col in required_exp_cols):
            logging.warning(f"Skipping experimental data plotting. DataFrame missing required columns: {', '.join(c for c in required_exp_cols if c not in experimental_data_df.columns)}")
            plot_df_exp_filtered = pd.DataFrame() # Empty df
        else:
            plot_df_exp_filtered = experimental_data_df[required_exp_cols].dropna().copy()
            if log_scale: plot_df_exp_filtered = plot_df_exp_filtered[pd.to_numeric(plot_df_exp_filtered[strain_rate_col], errors='coerce') > 0]

        if plot_df_exp_filtered.empty:
            logging.warning("No valid experimental data points found for multi-Wilkerson plot.")
        else:
             # Apply error filtering if needed
            if filter_high_error_perc is not None and spall_unc_col and spall_unc_col in experimental_data_df.columns:
                 y_err = pd.to_numeric(experimental_data_df.loc[plot_df_exp_filtered.index, spall_unc_col].abs(), errors='coerce')
                 if y_err is not None:
                     with np.errstate(divide='ignore', invalid='ignore'):
                         relative_error = np.abs(y_err / plot_df_exp_filtered[spall_col].replace(0, np.nan)) * 100
                     valid_points_mask = relative_error <= filter_high_error_perc
                     valid_points_mask = valid_points_mask | relative_error.isna()
                     if not valid_points_mask.all():
                         removed_count = sum(~valid_points_mask)
                         logging.info(f"  Filtering experimental data: Removing {removed_count} points with >{filter_high_error_perc:.0f}% relative Y error.")
                         plot_df_exp_filtered = plot_df_exp_filtered[valid_points_mask]

            if not plot_df_exp_filtered.empty:
                for material, group in plot_df_exp_filtered.groupby(material_col):
                    material_label = utils.MATERIAL_MAPPING.get(material, material)
                    color = color_map.get(material_label, 'black')
                    marker = marker_map.get(material_label, 'o')
                    if not group.empty:
                        # Plot exp data against STRAIN RATE
                        handle = ax.scatter(group[strain_rate_col], group[spall_col],
                                            label=f'{material_label}', # Label for legend grouping
                                            color=color, marker=marker, s=50, alpha=0.8, zorder=5)
                        if group_by_material and material_label not in exp_handles:
                             exp_handles[material_label] = handle # Store handle by material label
                        plotted_exp_materials.append(material_label)
                    else:
                         logging.debug(f"No valid experimental data for material '{material_label}' after filtering.")
                logging.info(f" Plotted experimental data for: {list(set(plotted_exp_materials))}")
            else:
                logging.warning("No valid experimental data points remain after filtering.")

        # --- Plot Literature Data (vs Strain Rate) ---
        if literature_data_file:
            logging.debug(f"Attempting to plot literature data from: {literature_data_file}")
            lit_req_cols = [strain_rate_col, spall_col, 'Source']
            lit_df = load_literature_data(literature_data_file, required_columns=lit_req_cols)
            if lit_df is not None:
                plot_lit_df = lit_df.dropna(subset=[spall_col, strain_rate_col]).copy()
                if log_scale: plot_lit_df = plot_lit_df[pd.to_numeric(plot_lit_df[strain_rate_col], errors='coerce') > 0]

                if plot_lit_df.empty:
                     logging.warning("No valid literature data points found after filtering.")
                else:
                    sources = plot_lit_df['Source'].unique()
                    prop_cycle = plt.rcParams['axes.prop_cycle']
                    lit_colors = prop_cycle.by_key()['color']
                    logging.info(f"Plotting literature data for sources: {', '.join(sources)}")
                    for idx, source in enumerate(sources):
                        source_data = plot_lit_df[plot_lit_df['Source'] == source]
                        if source_data.empty: continue
                        logging.debug(f"  Plotting literature source: {source} ({len(source_data)} points)")
                        lit_x = source_data[strain_rate_col]
                        lit_y = source_data[spall_col]
                        lit_marker = marker_map.get(source, marker_map.get("Default", "d"))
                        # Offset color cycle index
                        color_cycle_idx = (len(plotted_exp_materials) + len(grain_sizes_dict) + idx) % len(lit_colors)
                        lit_color = color_map.get(source, lit_colors[color_cycle_idx])
                        # Plot lit data against STRAIN RATE
                        handle = ax.scatter(lit_x, lit_y, s=80, alpha=0.6, color=lit_color, label=source, marker=lit_marker, edgecolors='grey', zorder=4)
                        lit_handles[source] = handle
                    logging.info(" Plotted literature data.")
            else:
                logging.warning(f"Could not load or process literature data from {literature_data_file}.")

        # --- Calculate and Plot Multiple Wilkerson Models (PRL Version) ---
        logging.info("Calculating Wilkerson model predictions (PRL paper) for multiple grain sizes...")
        try:
            required_wp_keys = ['sigma0_pa', 'ky_sqrtm', 'E_pa', 'Reos_pa', 'K0_pa', 'rho', 'N2', 'N0_GB', 'd0_G']
            if not all(key in wilkerson_params_base for key in required_wp_keys):
                missing_keys = [key for key in required_wp_keys if key not in wilkerson_params_base]
                logging.error(f"Missing required base Wilkerson parameters for PRL model: {missing_keys}. Skipping model lines.")
            else:
                # Generate a range of SPALL STRENGTHS (y-axis)
                min_spall_gpa = max(0.1, ylim[0] if ylim and ylim[0] > 0 else 0.1)
                max_spall_gpa = ylim[1] if ylim and ylim[1] > min_spall_gpa else 25
                num_points = 200
                model_spall_strengths_gpa = np.linspace(min_spall_gpa, max_spall_gpa, num_points)
                model_spall_strengths_pa = model_spall_strengths_gpa * 1e9

                default_linestyles = ['-', '--', ':', '-.']
                if model_linestyle_map is None: model_linestyle_map = {}
                base_params = wilkerson_params_base.copy()

                for i, (label, dG_val) in enumerate(grain_sizes_dict.items()):
                    logging.info(f"  Calculating PRL model for {label} (dG = {dG_val:.2e} m)...")
                    current_params = base_params

                    try:
                        # Call the NEW function
                        dG_input = dG_val
                        if dG_val <= 0 or np.isinf(dG_val): dG_input = np.inf # Handle SC

                        expansion_rates = calculate_expansion_rate_PRL(
                            Sigma_m_star_Pa=model_spall_strengths_pa,
                            dG=dG_input,
                            params=current_params
                        )

                        # Filter out NaNs or non-positive rates before plotting
                        valid_mask = pd.notna(expansion_rates) & (expansion_rates > 1e-9)
                        plot_expansion_rates = expansion_rates[valid_mask]
                        plot_spall_strengths_gpa = model_spall_strengths_gpa[valid_mask]

                        if len(plot_expansion_rates) > 0:
                            linestyle = model_linestyle_map.get(label, default_linestyles[i % len(default_linestyles)])
                            # Use distinct colors for model lines
                            model_color = plt.cm.viridis(i / max(1, len(grain_sizes_dict) - 1)) if len(grain_sizes_dict) > 1 else 'black'

                            logging.debug(f"    Plotting model line (PRL) for {label} with style '{linestyle}' and color '{model_color}'")
                            logging.debug(f"      Data points (first 5): X(ExpRate)={plot_expansion_rates[:5]}, Y(SpallGPa)={plot_spall_strengths_gpa[:5]}")
                            # Plot EXPANSION RATE (x) vs SPALL STRENGTH (y)
                            model_line, = ax.plot(plot_expansion_rates,
                                                  plot_spall_strengths_gpa,
                                                  color=model_color, linestyle=linestyle, linewidth=2.5, label=f'{label} (PRL Model)', zorder=10)
                            model_handles[f'{label} (PRL Model)'] = model_line
                            model_lines_plotted = True
                            logging.info(f"    Plotted model line (PRL) for {label}.")
                        else:
                            logging.warning(f"    Wilkerson model calculation (PRL) failed or yielded no valid points for {label}. Line not plotted.")
                    except Exception as model_calc_e_inner:
                         logging.exception(f"  Error calculating Wilkerson model (PRL) for {label}: {model_calc_e_inner}. Skipping this line.")

        except Exception as model_calc_e_outer:
            logging.exception(f"Error during multi-Wilkerson model calculation setup (PRL): {model_calc_e_outer}. Model lines may be missing.")

        # --- *** COMMENTED OUT OLD WILKERSON MODEL BLOCK *** ---
        # logging.info("Calculating Complex Wilkerson model predictions for multiple grain sizes...")
        # try:
        #     # ... (code for old model calculation and plotting) ...
        # except Exception as old_model_loop_e:
        #     logging.exception(f"  Error calculating OLD Wilkerson model for {label}: {old_model_loop_e}")
        # --- *** END OF COMMENTED OUT BLOCK *** ---

        # --- Customize Plot ---
        logging.debug("Customizing plot axes and labels...")
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        if log_scale: ax.set_xscale('log')
        if xlim: ax.set_xlim(xlim) # Ensure xlim is appropriate for expansion rate
        if ylim: ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major')
        ax.grid(True, which='both' if log_scale else 'major', linestyle='--', alpha=0.6)

        # --- Create Legend ---
        logging.debug("Creating legend...")
        all_handles = []
        all_labels = []
        if group_by_material:
            sorted_materials = sorted(exp_handles.keys())
            for material_label in sorted_materials:
                all_handles.append(exp_handles[material_label])
                all_labels.append(f"{material_label} Exp.")
        sorted_lit_sources = sorted(lit_handles.keys())
        for source in sorted_lit_sources:
            all_handles.append(lit_handles[source])
            all_labels.append(source)
        sorted_model_labels = sorted(model_handles.keys())
        for label in sorted_model_labels:
            all_handles.append(model_handles[label])
            all_labels.append(label)
        # *** CORRECTED: Use the show_legend argument passed to the function ***
        if show_legend and all_handles:
            ax.legend(handles=all_handles, labels=all_labels, loc='best', fontsize=14)
        else:
            logging.info("No elements found to include in the legend or legend disabled.")

        # --- Save Plot ---
        try:
            plt.tight_layout()
            plt.savefig(output_filename, dpi=150)
            logging.info(f"Successfully saved Multi-Wilkerson plot to: {output_filename}")
            plot_saved = True
        except Exception as e:
            logging.exception(f"Error saving Multi-Wilkerson plot {output_filename}: {e}")

    except Exception as e:
        logging.exception(f"Error occurred during generation of plot {output_filename}: {e}")
    finally:
        logging.debug(f"Closing figure for {output_filename}")
        plt.close(fig)
        if not plot_saved:
             logging.error(f"Plot generation failed or was skipped, plot NOT saved: {output_filename}")


# --- Combined Mean Raw Velocity Plot ---
def plot_combined_mean_traces(
    mean_trace_files, output_filename, title="Combined Mean Velocity Traces",
    xlabel="Time (ns)", ylabel="Mean Free Surface Velocity (m/s)",
    xlim=None, ylim=None, legend_mapping=utils.ENERGY_VELOCITY_MAPPING,
    material_map=utils.MATERIAL_MAPPING, color_map=utils.COLOR_MAPPING,
    plot_std_dev=True, std_dev_alpha=0.2, show_legend=True # Added show_legend
    ):
    """ Plots combined mean velocity traces with std dev shading. """
    logging.debug(f"Starting plot_combined_mean_traces for {output_filename}")
    fig, ax = plt.subplots()
    plotted_labels = set()
    plot_saved = False
    if not mean_trace_files:
        logging.warning("No mean trace files provided for combined plot.")
        plt.close(fig)
        return

    try:
        for idx, file in enumerate(mean_trace_files):
            base_name = os.path.basename(file)
            bin_name = base_name.replace('_mean_raw_velocity.csv', '')
            logging.info(f"  Plotting mean trace for bin: {bin_name}")
            try:
                data = pd.read_csv(file)
                required_cols = ['Time (ns)', 'Mean Velocity (m/s)']
                if plot_std_dev: required_cols.append('Std Dev Velocity (m/s)')
                if not all(col in data.columns for col in required_cols):
                    logging.warning(f"Skipping {base_name}: Missing required columns ({required_cols})."); continue
                time = data['Time (ns)']
                mean_velocity = data['Mean Velocity (m/s)']
                velocity_std = data['Std Dev Velocity (m/s)'] if plot_std_dev else None

                material_type, energy_label, _ = utils.extract_legend_info(bin_name, material_map, legend_mapping)
                label = f"{material_type} ({energy_label})" if material_type and energy_label else bin_name
                color = color_map.get(material_type, plt.rcParams['axes.prop_cycle'].by_key()['color'][idx % 10])
                logging.debug(f"    Plotting {label} with color {color}")

                ax.plot(time, mean_velocity, linestyle='-', color=color, label=label, linewidth=plt.rcParams['lines.linewidth'])
                plotted_labels.add(label)

                if velocity_std is not None and pd.notna(velocity_std).all():
                    ax.fill_between(time, mean_velocity - velocity_std, mean_velocity + velocity_std, color=color, alpha=std_dev_alpha, edgecolor='none')
                elif plot_std_dev:
                    logging.warning(f"Std deviation contains NaNs or invalid values for {bin_name}. Shading skipped.")

            except FileNotFoundError: logging.warning(f"Mean trace file not found: {file}")
            except pd.errors.EmptyDataError: logging.warning(f"Skipping empty mean trace file: {base_name}")
            except Exception as e: logging.exception(f"Error plotting mean trace file {base_name}: {e}")

        logging.debug("Customizing plot axes and labels...")
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        if xlim: ax.set_xlim(xlim);
        if ylim: ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major'); ax.grid(True, which='major', linestyle='--', alpha=0.6)
        # *** CORRECTED: Use the show_legend argument passed to the function ***
        if show_legend and plotted_labels: ax.legend(loc="best", fontsize=14)
        try:
            plt.tight_layout()
        except Exception as tle:
            logging.warning(f"tight_layout failed for {output_filename}: {tle}")

        logging.info(f"Attempting to save plot: {output_filename}")
        plt.savefig(output_filename, dpi=150)
        logging.info(f"Successfully saved combined mean raw traces plot to: {output_filename}")
        plot_saved = True

    except Exception as e:
        logging.exception(f"Error occurred during generation of plot {output_filename}: {e}")
    finally:
        logging.debug(f"Closing figure for {output_filename}")
        plt.close(fig)
        if not plot_saved:
             logging.error(f"Plot generation failed or was skipped, plot NOT saved: {output_filename}")


# --- Combined Mean Model Plot ---
def plot_combined_mean_models(
    mean_model_files, output_filename, title="Combined Mean 5-Segment Models",
    xlabel="Time (ns)", ylabel="Mean Model Velocity (m/s)",
    xlim=None, ylim=None, legend_mapping=utils.ENERGY_VELOCITY_MAPPING,
    material_map=utils.MATERIAL_MAPPING, color_map=utils.COLOR_MAPPING,
    plot_std_dev=True, std_dev_alpha=0.2, show_legend=True # Added show_legend
    ):
    """ Plots combined mean fitted model traces with std dev shading. """
    logging.debug(f"Starting plot_combined_mean_models for {output_filename}")
    fig, ax = plt.subplots()
    plotted_labels = set()
    plot_saved = False
    if not mean_model_files:
        logging.warning("No mean model files provided for combined plot.")
        plt.close(fig)
        return

    try:
        for idx, file in enumerate(mean_model_files):
            base_name = os.path.basename(file)
            bin_name = base_name.replace('_mean_model_velocity.csv', '')
            logging.info(f"  Plotting mean model for bin: {bin_name}")
            try:
                data = pd.read_csv(file)
                required_cols = ['Time (ns)', 'Mean Model Velocity (m/s)']
                if plot_std_dev: required_cols.append('Std Dev Model Velocity (m/s)')
                if not all(col in data.columns for col in required_cols):
                    logging.warning(f"Skipping {base_name}: Missing required columns ({required_cols})."); continue
                time = data['Time (ns)']
                mean_velocity = data['Mean Model Velocity (m/s)']
                velocity_std = data['Std Dev Model Velocity (m/s)'] if plot_std_dev else None

                material_type, energy_label, _ = utils.extract_legend_info(bin_name, material_map, legend_mapping)
                label = f"{material_type} ({energy_label})" if material_type and energy_label else bin_name
                color = color_map.get(material_type, plt.rcParams['axes.prop_cycle'].by_key()['color'][idx % 10])
                logging.debug(f"    Plotting {label} with color {color}")

                ax.plot(time, mean_velocity, linestyle='--', color=color, label=label, linewidth=plt.rcParams['lines.linewidth'])
                plotted_labels.add(label)

                if velocity_std is not None and pd.notna(velocity_std).all():
                    ax.fill_between(time, mean_velocity - velocity_std, mean_velocity + velocity_std, color=color, alpha=std_dev_alpha, edgecolor='none')
                elif plot_std_dev:
                    logging.warning(f"Std deviation contains NaNs or invalid values for {bin_name} model. Shading skipped.")

            except FileNotFoundError: logging.warning(f"Mean model file not found: {file}")
            except pd.errors.EmptyDataError: logging.warning(f"Skipping empty mean model file: {base_name}")
            except Exception as e: logging.exception(f"Error plotting mean model file {base_name}: {e}")

        logging.debug("Customizing plot axes and labels...")
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        if xlim: ax.set_xlim(xlim);
        if ylim: ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major'); ax.grid(True, which='major', linestyle='--', alpha=0.6)
        # *** CORRECTED: Use the show_legend argument passed to the function ***
        if show_legend and plotted_labels: ax.legend(loc="best", fontsize=14)
        try:
            plt.tight_layout()
        except Exception as tle:
            logging.warning(f"tight_layout failed for {output_filename}: {tle}")

        logging.info(f"Attempting to save plot: {output_filename}")
        plt.savefig(output_filename, dpi=150)
        logging.info(f"Successfully saved combined mean model plot to: {output_filename}")
        plot_saved = True

    except Exception as e:
        logging.exception(f"Error occurred during generation of plot {output_filename}: {e}")
    finally:
        logging.debug(f"Closing figure for {output_filename}")
        plt.close(fig)
        if not plot_saved:
             logging.error(f"Plot generation failed or was skipped, plot NOT saved: {output_filename}")


# --- Plotting Model Predictions (2D) ---
def plot_model_vs_feature(
    data_df, x_col, y_actual_col, y_pred_col, output_filename,
    material_col='Material', title='Model Prediction vs. Feature',
    xlabel=None, ylabel=None, log_x=False, xlim=None, ylim=None,
    color_map=None, marker_map=None, show_legend=True # Added show_legend
    ):
    """ Plots actual vs. predicted values against a specific feature, grouped by material. """
    if color_map is None: color_map = utils.COLOR_MAPPING
    if marker_map is None: marker_map = utils.MARKER_MAPPING
    logging.debug(f"Starting plot_model_vs_feature for {output_filename}")
    fig, ax = plt.subplots()
    plotted_materials = set(); handles = []; labels = []
    plot_saved = False

    xlabel = xlabel or x_col
    ylabel = ylabel or y_pred_col

    try:
        required_cols = [x_col, y_actual_col, y_pred_col, material_col]
        if not all(c in data_df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in data_df.columns]
            logging.error(f"Plotting model vs feature failed for {output_filename}: DataFrame missing required columns: {missing}")
            plt.close(fig); return

        plot_data = data_df[required_cols].dropna().copy()
        logging.debug(f"Model plot DF shape after NaN drop: {plot_data.shape}. Columns: {plot_data.columns.tolist()}")

        if plot_data.empty:
            logging.warning(f"No valid data points found for plotting model predictions vs {x_col}.")
        else:
            for material_label, group_data in plot_data.groupby(material_col):
                logging.debug(f"  Plotting model predictions for group: '{material_label}' ({len(group_data)} points)")
                color = color_map.get(material_label, color_map.get("Default", "gray"))
                marker_actual = marker_map.get(material_label, marker_map.get("Default", "o"))
                marker_pred = 'x'

                h_actual = ax.scatter(group_data[x_col], group_data[y_actual_col],
                                      marker=marker_actual, color=color, alpha=0.7, s=60,
                                      label=f'{material_label} Actual')
                h_pred = ax.scatter(group_data[x_col], group_data[y_pred_col],
                                    marker=marker_pred, color=color, alpha=0.7, s=60,
                                    label=f'{material_label} Predicted')

                if material_label not in plotted_materials:
                    handles.extend([h_actual, h_pred])
                    labels.extend([f'{material_label} Actual', f'{material_label} Predicted'])
                    plotted_materials.add(material_label)

        logging.debug("Customizing model plot axes and labels...")
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        if log_x: ax.set_xscale('log')
        if xlim: ax.set_xlim(xlim);
        if ylim: ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major')
        ax.grid(True, which='major' if not log_x else 'both', linestyle='--', alpha=0.6)

        logging.debug("Creating model plot legend...")
        # *** CORRECTED: Use the show_legend argument passed to the function ***
        if show_legend and handles: ax.legend(handles=handles, labels=labels, loc='best', fontsize=14)
        else: logging.info("No data plotted or legend disabled, legend skipped.")
        try:
            plt.tight_layout()
        except Exception as tle:
            logging.warning(f"tight_layout failed for {output_filename}: {tle}")

        logging.info(f"Attempting to save plot: {output_filename}")
        plt.savefig(output_filename, dpi=150)
        logging.info(f"Successfully saved model prediction plot to: {output_filename}")
        plot_saved = True

    except Exception as e:
        logging.exception(f"Error occurred during generation of plot {output_filename}: {e}")
    finally:
        logging.debug(f"Closing figure for {output_filename}")
        plt.close(fig)
        if not plot_saved:
             logging.error(f"Plot generation failed or was skipped, plot NOT saved: {output_filename}")


# --- 3D Model Surface Plot ---
def plot_model_3d_surface(
    model, scaler, X_original_df, y_actual,
    feature1_name, feature2_name, all_feature_names,
    strain_rate_original_col, # Added argument
    output_filename, material_col='Material',
    title='Model Surface Plot', xlabel=None, ylabel=None, zlabel=None,
    color_map=None, marker_map=None, grid_points=25
    ):
    """ Generates a 3D surface plot for a trained regression model. """
    if color_map is None: color_map = utils.COLOR_MAPPING
    if marker_map is None: marker_map = utils.MARKER_MAPPING
    logging.debug(f"Starting plot_model_3d_surface for {output_filename}")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_saved = False

    xlabel = xlabel or feature1_name
    ylabel = ylabel or feature2_name
    zlabel = zlabel or y_actual.name or 'Predicted Value'

    try:
        # Check for required columns in the original DataFrame
        required_cols_orig = [feature1_name, feature2_name, material_col] + all_feature_names
        if strain_rate_original_col not in X_original_df.columns:
             logging.warning(f"Original strain rate column '{strain_rate_original_col}' not found in X_original_df, needed for potential feature filling.")
             # Decide if this is critical or can proceed with other means
        missing_cols = [c for c in required_cols_orig if c not in X_original_df.columns]
        if missing_cols:
            logging.error(f"Cannot generate 3D plot: X_original_df missing columns: {missing_cols}")
            plt.close(fig); return

        # Check index alignment and emptiness
        if not y_actual.index.equals(X_original_df.index):
             logging.error(f"Cannot generate 3D plot: X_original_df and y_actual indices do not match.")
             plt.close(fig); return
        if X_original_df.empty or y_actual.empty:
             logging.warning(f"Skipping 3D plot generation for {output_filename}: Input data is empty.")
             plt.close(fig); return

        # Prepare data for plotting (only valid points for axes limits and scatter)
        plot_data_df = X_original_df.loc[y_actual.dropna().index].copy() # Select rows with non-NaN target
        plot_data_df['__target__'] = y_actual # Add target for convenience
        plot_data_df.dropna(subset=[feature1_name, feature2_name, '__target__', material_col], inplace=True)

        if plot_data_df.empty:
            logging.warning(f"Skipping 3D plot generation for {output_filename}: No valid data points after dropping NaNs for plotting.")
            plt.close(fig); return

        # Determine grid limits from the valid plotting data
        x1_min, x1_max = plot_data_df[feature1_name].min(), plot_data_df[feature1_name].max()
        x2_min, x2_max = plot_data_df[feature2_name].min(), plot_data_df[feature2_name].max()
        if pd.isna(x1_min) or pd.isna(x1_max) or pd.isna(x2_min) or pd.isna(x2_max):
             logging.warning(f"Skipping 3D plot for {output_filename}: NaN values found in plot feature limits.")
             plt.close(fig); return
        x1_margin = (x1_max - x1_min) * 0.05 if x1_max > x1_min else 0.1
        x2_margin = (x2_max - x2_min) * 0.05 if x2_max > x2_min else 0.1
        x1_lin = np.linspace(x1_min - x1_margin, x1_max + x1_margin, grid_points)
        x2_lin = np.linspace(x2_min - x2_margin, x2_max + x2_margin, grid_points)
        x1_grid, x2_grid = np.meshgrid(x1_lin, x2_lin)

        # Prepare the full feature grid for prediction
        grid_features_df_for_pred = pd.DataFrame({
            feature1_name: x1_grid.ravel(),
            feature2_name: x2_grid.ravel()
        })

        # Fill other features needed by the model using means from the original data FOR THIS MATERIAL
        material_data_for_means = X_original_df.loc[y_actual.dropna().index, all_feature_names]
        if material_data_for_means.empty:
             logging.error(f"Cannot calculate feature means for grid: No valid training data points found for {output_filename}.")
             plt.close(fig); return
        feature_means = material_data_for_means.mean()
        logging.debug(f"Feature means for 3D plot grid: {feature_means.to_dict()}")

        all_features_present = True
        for f_name in all_feature_names:
            if f_name not in grid_features_df_for_pred.columns:
                mean_val = feature_means.get(f_name)
                if pd.isna(mean_val):
                     logging.warning(f"Mean for feature '{f_name}' is NaN. Using 0 for grid prediction.")
                     mean_val = 0
                grid_features_df_for_pred[f_name] = mean_val
        # Ensure correct column order
        grid_features_df_for_pred = grid_features_df_for_pred[all_feature_names]

        # Scale the grid features using the provided scaler
        if scaler is not None:
             try:
                 grid_features_scaled = scaler.transform(grid_features_df_for_pred)
                 # Convert back to DataFrame to keep feature names for prediction if model requires it
                 grid_features_scaled_df = pd.DataFrame(grid_features_scaled, columns=all_feature_names)
                 features_to_predict = grid_features_scaled_df
             except Exception as scale_err:
                 logging.exception(f"Error scaling grid features: {scale_err}")
                 plt.close(fig); return
        else:
             features_to_predict = grid_features_df_for_pred # Predict on original features if no scaler


        logging.debug("Predicting Z values for the surface grid...")
        try:
            Z_grid = model.predict(features_to_predict).reshape(x1_grid.shape)
        except Exception as pred_err:
             logging.exception(f"Error predicting on grid: {pred_err}. Cannot generate surface.")
             plt.close(fig); return

        logging.debug("Plotting 3D surface...")
        surf = ax.plot_surface(x1_grid, x2_grid, Z_grid, cmap='viridis', alpha=0.6, edgecolor='none', label='Model Prediction')

        handles = []; labels = []
        # --- Scatter Plot Actual Data (Optional - uncomment if needed) ---
        # logging.debug("Plotting actual data points...")
        # material_label = plot_data_df[material_col].iloc[0] # Get material label
        # color = color_map.get(material_label, color_map.get("Default", "gray"))
        # marker = marker_map.get(material_label, marker_map.get("Default", "o"))
        # try:
        #      h = ax.scatter(plot_data_df[feature1_name], plot_data_df[feature2_name], plot_data_df['__target__'],
        #                     color=color, marker=marker, s=50, alpha=0.8, label=f'{material_label} Actual', depthshade=True)
        #      handles.append(h); labels.append(f'{material_label} Actual')
        # except Exception as scatter_err: logging.warning(f"Could not plot scatter points for {material_label}: {scatter_err}")

        logging.debug("Customizing 3D plot axes and labels...")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
        ax.set_title(title, fontsize=plt.rcParams['figure.titlesize']*0.9, pad=20)
        if handles: ax.legend(handles=handles, labels=labels, loc='best', fontsize=plt.rcParams['legend.fontsize']*0.8)
        elif surf is not None:
             # Add color bar if only surface is plotted
             try: fig.colorbar(surf, shrink=0.5, aspect=10, label=zlabel)
             except Exception as cb_err: logging.warning(f"Could not add colorbar: {cb_err}")
             # ax.legend(handles=[surf], labels=['Model Prediction'], loc='best', fontsize=plt.rcParams['legend.fontsize']*0.8) # Legend for surface is tricky

        logging.info(f"Attempting to save plot: {output_filename}")
        try: plt.tight_layout()
        except ValueError: logging.warning("tight_layout failed, continuing without it.")
        plt.savefig(output_filename, dpi=150)
        logging.info(f"Successfully saved 3D Model plot to: {output_filename}")
        plot_saved = True

    except Exception as e:
        logging.exception(f"Error occurred during generation of 3D plot {output_filename}: {e}")
    finally:
        logging.debug(f"Closing figure for {output_filename}")
        plt.close(fig)
        if not plot_saved:
             logging.error(f"Plot generation failed or was skipped, plot NOT saved: {output_filename}")


# --- Combined 3D Material Surface Plot ---
def plot_combined_material_surfaces(
    trained_models_dict, data_df, feature_cols, target_col, material_col,
    plot_feature1, plot_feature2, strain_rate_original_col, output_filename,
    title='Combined Model Surfaces', xlabel=None, ylabel=None, zlabel=None,
    color_map=None, marker_map=None, grid_points=50,
    surface_alpha=0.6, scatter_alpha=0.7, scatter_size=60, show_legend=True # Added show_legend
    ):
    """ Generates a single 3D plot showing multiple fitted model surfaces. """
    if color_map is None: color_map = utils.COLOR_MAPPING
    if marker_map is None: marker_map = utils.MARKER_MAPPING
    logging.info(f"Starting combined 3D surface plot: {output_filename}")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_saved = False
    plotted_handles = []
    plotted_labels = []

    xlabel = xlabel or plot_feature1
    ylabel = ylabel or plot_feature2
    zlabel = zlabel or target_col

    try:
        materials_to_plot = list(trained_models_dict.keys())
        if not materials_to_plot:
            logging.error("No materials/models provided in trained_models_dict. Cannot generate combined plot.")
            plt.close(fig); return

        # Ensure all necessary columns are present in the main DataFrame
        required_df_cols = [plot_feature1, plot_feature2, target_col, material_col] + feature_cols
        if strain_rate_original_col not in data_df.columns:
             logging.warning(f"Original strain rate column '{strain_rate_original_col}' not in data_df, needed for feature filling if scaling was used.")
        missing_df_cols = [col for col in required_df_cols if col not in data_df.columns]
        if missing_df_cols:
            logging.error(f"Combined data_df missing required columns: {missing_df_cols}. Cannot generate combined plot.")
            plt.close(fig); return

        # Filter data for the materials being plotted and drop NaNs for axis limits
        plot_data_all = data_df[data_df[material_col].isin(materials_to_plot)].copy()
        plot_data_all.dropna(subset=[plot_feature1, plot_feature2, target_col], inplace=True)

        if plot_data_all.empty:
            logging.warning(f"No valid data points found for materials {materials_to_plot} after NaN drop. Skipping combined plot.")
            plt.close(fig); return

        # Determine overall grid range
        x_plot_vals = plot_data_all[plot_feature1]
        y_plot_vals = plot_data_all[plot_feature2]
        x_min, x_max = x_plot_vals.min(), x_plot_vals.max()
        y_min, y_max = y_plot_vals.min(), y_plot_vals.max()

        if pd.isna(x_min) or pd.isna(x_max) or pd.isna(y_min) or pd.isna(y_max):
             logging.error(f"Cannot generate grid: NaN values found in combined plot feature limits.")
             plt.close(fig); return

        x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
        y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        x_lin = np.linspace(x_min - x_margin, x_max + x_margin, grid_points)
        y_lin = np.linspace(y_min - y_margin, y_max + y_margin, grid_points)
        X_grid, Y_grid = np.meshgrid(x_lin, y_lin)

        # Use a colormap for surfaces if not provided by color_map
        surface_cmaps = plt.cm.viridis(np.linspace(0, 1, len(materials_to_plot)))

        for idx, material_label in enumerate(materials_to_plot):
            logging.info(f" Processing surface and points for: {material_label}")
            if material_label not in trained_models_dict:
                logging.warning(f" No model info found for '{material_label}'. Skipping.")
                continue

            model_info = trained_models_dict[material_label]
            model = model_info.get('model')
            scaler = model_info.get('scaler') # Get the scaler used for this material
            X_orig_mat = model_info.get('X_original') # Original data for THIS material
            y_actual_mat = model_info.get('y_actual') # Actual target for THIS material

            if model is None or scaler is None or X_orig_mat is None or y_actual_mat is None:
                logging.warning(f" Model, scaler, X_original, or y_actual missing for '{material_label}'. Skipping.")
                continue

            # Prepare grid features specific to this material
            grid_features_df_for_pred = pd.DataFrame({
                plot_feature1: X_grid.ravel(),
                plot_feature2: Y_grid.ravel()
            })
            # Use means from THIS material's original training data to fill other features
            feature_means = X_orig_mat[feature_cols].mean()
            all_features_ok = True
            for f_name in feature_cols:
                if f_name not in grid_features_df_for_pred.columns:
                    mean_val = feature_means.get(f_name, 0)
                    if pd.isna(mean_val): mean_val = 0; logging.warning(f" Mean for feature '{f_name}' is NaN for {material_label}. Using 0.")
                    grid_features_df_for_pred[f_name] = mean_val
            grid_features_df_for_pred = grid_features_df_for_pred[feature_cols] # Ensure order

            # Scale grid features using the correct scaler
            try:
                grid_features_scaled = scaler.transform(grid_features_df_for_pred)
                features_to_predict = pd.DataFrame(grid_features_scaled, columns=feature_cols)
            except Exception as scale_err:
                 logging.exception(f"  Error scaling grid features for {material_label}: {scale_err}. Skipping surface.")
                 continue

            # Predict on scaled grid
            logging.debug(f"  Predicting Z values for '{material_label}' surface...")
            try:
                Z_grid = model.predict(features_to_predict).reshape(X_grid.shape)
            except Exception as pred_err:
                 logging.exception(f"  Error predicting on grid for {material_label}: {pred_err}. Skipping surface.")
                 continue

            # Plot Surface
            logging.debug(f"  Plotting surface for {material_label}...")
            color = color_map.get(material_label, surface_cmaps[idx]) # Use specific color or cycle
            surf = ax.plot_surface(X_grid, Y_grid, Z_grid, color=color, alpha=surface_alpha,
                                   rstride=1, cstride=1, linewidth=0.1, edgecolors='grey')
            # Create proxy artist for legend (use a patch)
            proxy = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=surface_alpha)
            plotted_handles.append(proxy)
            plotted_labels.append(f'{material_label} Model')

            # Plot Scatter Points for this material (optional)
            logging.debug(f"  Plotting scatter points for {material_label}...")
            material_data_for_scatter = X_orig_mat.copy()
            material_data_for_scatter[target_col] = y_actual_mat
            material_data_for_scatter.dropna(subset=[plot_feature1, plot_feature2, target_col], inplace=True)
            if not material_data_for_scatter.empty:
                marker = marker_map.get(material_label, marker_map.get("Default", "o"))
                scatter = ax.scatter(material_data_for_scatter[plot_feature1], material_data_for_scatter[plot_feature2], material_data_for_scatter[target_col],
                                     color=color, marker=marker, s=scatter_size, alpha=scatter_alpha,
                                     label=f'{material_label} Data', edgecolors='k', depthshade=True)
                plotted_handles.append(scatter)
                plotted_labels.append(f'{material_label} Data')

        # --- Final Plot Customization ---
        logging.debug("Customizing combined plot axes and labels...")
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_zlabel(zlabel, fontsize=20)
        ax.set_title(title, fontsize=plt.rcParams['figure.titlesize']*0.9)

        # *** CORRECTED: Use the show_legend argument passed to the function ***
        if show_legend and plotted_handles:
            ax.legend(handles=plotted_handles, labels=plotted_labels, loc='best', fontsize=14)
        try:
            plt.tight_layout()
        except Exception as tle:
            logging.warning(f"tight_layout failed for {output_filename}: {tle}")

        # --- Save Plot ---
        logging.info(f"Attempting to save combined plot: {output_filename}")
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        logging.info(f"Successfully saved combined 3D plot to: {output_filename}")
        plot_saved = True

    except Exception as e:
        logging.exception(f"Error occurred during generation of combined 3D plot {output_filename}: {e}")
    finally:
        logging.debug(f"Closing figure for {output_filename}")
        plt.close(fig)
        if not plot_saved:
             logging.error(f"Combined 3D plot generation failed or was skipped, plot NOT saved: {output_filename}")


# --- Placeholder for plot_elastic_net_results ---
def plot_elastic_net_results(*args, **kwargs):
    logging.warning("plot_elastic_net_results function is a placeholder and does nothing.")
    pass

def plot_interactive_spall_vs_strain_rate(
    df,
    output_filename,
    grain_sizes_dict,
    wilkerson_params_base,
    strain_rate_col='Strain Rate (s^-1)',
    spall_col='Spall Strength (GPa)',
    spall_unc_col='Spall Strength Err (GPa)',
    material_col='Material',
    filter_high_error_perc=100,
    title="Interactive Spall Strength vs. Strain Rate"
):
    """
    Creates an interactive HTML plot of spall strength vs strain rate using plotly.
    
    Args:
        df (pd.DataFrame): DataFrame containing experimental data
        output_filename (str): Path to save the HTML file
        grain_sizes_dict (dict): Dictionary mapping grain size labels to values
        wilkerson_params_base (dict): Base parameters for Wilkerson model
        strain_rate_col (str): Column name for strain rate data
        spall_col (str): Column name for spall strength data
        spall_unc_col (str): Column name for spall strength uncertainty
        material_col (str): Column name for material type
        filter_high_error_perc (float): Maximum allowed relative error percentage
        title (str): Plot title
    """
    try:
        import plotly.graph_objects as go
        from datetime import datetime
        from .models import calculate_wilkerson_spall_complex
        
        # Create filtered DataFrame for plotting
        plot_df_exp_filtered = df[
            (df['Processing Status'] == 'Success') &
            (df[spall_unc_col] / df[spall_col] <= filter_high_error_perc / 100)
        ].copy()
        
        # Create interactive plot
        fig = go.Figure()
        
        # Map material types to colors
        color_map = {
            'Nano': 'red',
            'Poly': 'blue',
            'SC [100]': 'green',
            'SC [110]': 'purple',
            'SC [111]': 'orange'
        }
        
        # Add experimental data points
        for material_type in plot_df_exp_filtered[material_col].unique():
            material_data = plot_df_exp_filtered[plot_df_exp_filtered[material_col] == material_type]
            
            fig.add_trace(go.Scatter(
                x=material_data[strain_rate_col],
                y=material_data[spall_col],
                mode='markers',
                name=f'{material_type} (Exp)',
                marker=dict(
                    color=color_map.get(material_type, 'gray'),
                    size=10,
                    line=dict(width=1, color='black')
                ),
                text=material_data.apply(lambda row: f"""
                    Material: {row[material_col]}<br>
                    Strain Rate: {row[strain_rate_col]:.2e} s^-1<br>
                    Spall Strength: {row[spall_col]:.2f} GPa<br>
                    Error: ±{row[spall_unc_col]:.2f} GPa<br>
                    Processing Status: {row['Processing Status']}<br>
                    File: {row['FileName']}
                """, axis=1),
                hoverinfo='text'
            ))
        
        # Add Wilkerson model curves
        for grain_size_label, grain_size in grain_sizes_dict.items():
            # Generate points for the model curve
            strain_rates = np.logspace(3, 9, 100)
            wilkerson_params = wilkerson_params_base.copy()
            wilkerson_params['dG'] = grain_size
            
            # Calculate spall strengths using the complex Wilkerson model
            spall_strengths = calculate_wilkerson_spall_complex(
                strain_rate=strain_rates,
                dG=grain_size,
                sigma0_pa=wilkerson_params['sigma0_pa'],
                ky_sqrtm=wilkerson_params['ky_sqrtm'],
                E_pa=wilkerson_params['E_pa'],
                Reos_pa=wilkerson_params['Reos_pa'],
                K0_pa=wilkerson_params['K0_pa'],
                rho=wilkerson_params['rho'],
                N2=wilkerson_params['N2'],
                N0_GB=wilkerson_params['N0_GB'],
                d0_G=wilkerson_params['d0_G']
            )
            
            # Create hover text for model curve
            model_hover_text = [
                f"""
                Model: Wilkerson Complex<br>
                Grain Size: {grain_size*1e6:.1f} μm<br>
                Strain Rate: {sr:.2e} s^-1<br>
                Spall Strength: {ss:.2f} GPa
                """
                for sr, ss in zip(strain_rates, spall_strengths)
            ]
            
            fig.add_trace(go.Scatter(
                x=strain_rates,
                y=spall_strengths,
                mode='lines',
                name=f'Wilkerson Model ({grain_size_label})',
                line=dict(
                    dash='solid',
                    width=2
                ),
                text=model_hover_text,
                hoverinfo='text',
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis=dict(
                title='Expansion Rate ($\dot{v}/v_0$) [s$^{-1}$]',
                titlefont=dict(size=20),
                tickfont=dict(size=20),
                type='log',
                range=[3, 9]
            ),
            yaxis=dict(
                title='Spall Strength (GPa)',
                titlefont=dict(size=20),
                tickfont=dict(size=20),
                range=[0, 8]
            ),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(size=14)
            ),
            template='plotly_white',
            font=dict(size=20),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Add interactive features
        fig.update_traces(
            marker=dict(
                line=dict(width=1, color='DarkSlateGrey')
            )
        )
        
        # Save as HTML file
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        fig.write_html(output_filename)
        logging.info(f"Interactive plot saved to: {output_filename}")
        
    except ImportError as e:
        logging.error(f"Failed to import required packages for interactive plot: {e}")
        print("Please install plotly using: pip install plotly")
    except Exception as e:
        logging.exception(f"Error generating interactive plot: {e}")
